import sys
import collections
import argparse
import os
import timeit
import logging
import json
import ase
import ase.io
import msgnet
import tensorflow as tf
import numpy as np
import densitymsg
from densityloader import ChargeDataLoader, LazyChargeDataLoader
from densityhandler import DensityDataHandler
from trainer import DensityOutputTrainer
from ase.neighborlist import NeighborList

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Train graph convolution network", fromfile_prefix_chars="@"
    )
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--use_train_queue", action="store_true")
    parser.add_argument("--use_lazy_loader", action="store_true")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--split_file", type=str, default=None)

    return parser.parse_args(arg_list)

def get_model(cutoff):
    embedding_size = 256

    model = densitymsg.DensityMsgPassing(
        embedding_shape=(len(ase.data.chemical_symbols), embedding_size),
        edge_feature_expand=[(0, 0.01, cutoff+1)],
        use_edge_updates=False,
        num_passes=6,
        hard_cutoff=cutoff,
        single_atom_reference_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "single_atom_reference/"),
        )

    return model

def split_by_file(filename, graph_objects):
    with open(filename, "r") as f:
        split_dict = json.load(f)

    splits = collections.defaultdict(list)

    for i, obj in enumerate(graph_objects):
        for key, val in split_dict.items():
            if i in val:
                splits[key].append(obj)
            elif hasattr(obj, "member") and obj.member.name in val:
                splits[key].append(obj)
            elif hasattr(obj, "filename") and obj.filename in val:
                splits[key].append(obj)

    return splits

def train_model(args, logs_path):
    if args.use_lazy_loader:
        LoaderClass = LazyChargeDataLoader
    else:
        LoaderClass = ChargeDataLoader

    if args.dataset.endswith(".txt"):
        # Text file contains list of datafiles
        with open(args.dataset, "r") as datasetfiles:
            filelist = [os.path.join(os.path.dirname(args.dataset), line.strip('\n')) for line in datasetfiles]
    else:
        filelist = [args.dataset]

    graph_obj_list = []
    for i, filename in enumerate(filelist):
        logging.debug("loading file %d/%d: %s" % (i+1, len(filelist), filename))
        densityloader = LoaderClass(filename, args.cutoff)
        graph_obj_list.extend(densityloader.load())


    if args.split_file:
        splits = split_by_file(args.split_file, graph_obj_list)
        if "train" in splits:
            train_handler = DensityDataHandler(splits["train"])
        if "validation" in splits:
            validation_handler = DensityDataHandler(splits["validation"])
    else:
        data_handler = DensityDataHandler(graph_obj_list)
        train_handler, _, validation_handler = data_handler.train_test_split(split_type="count", validation_size=10, test_size=10)

    if args.use_train_queue:
        train_handler.setup_train_queue()
        num_samples_train_metric = 10
        train_metrics_handler = DensityDataHandler([train_handler.graph_objects[i].decompress() for i in np.random.permutation(len(train_handler))[0:num_samples_train_metric]])
        validation_handler.graph_objects = [g.decompress() for g in validation_handler.graph_objects]
    else:
        train_metrics_handler = train_handler
        validation_handler.graph_objects = [g.decompress() for g in validation_handler.graph_objects]
        train_handler.graph_objects = [g.decompress() for g in train_handler.graph_objects]


    batch_size = 1

    model = get_model(args.cutoff)

    trainer = DensityOutputTrainer(model, train_handler, batch_size=batch_size, initial_lr=args.learning_rate)

    num_steps = int(1e7)
    start_step = 0
    log_interval = 10000

    best_val_mae = np.inf
    best_val_step = 0

    start_time = timeit.default_timer()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if args.load_model:
            if args.load_model.endswith(".meta"):
                checkpoint = args.load_model.replace(".meta", "")
                logging.info("loading model from %s", checkpoint)
                start_step = int(checkpoint.split("/")[-1].split("-")[-1])
                model.load(sess, checkpoint)
            else:
                checkpoint = tf.train.get_checkpoint_state(args.load_model)
                logging.info("loading model from %s", checkpoint)
                start_step = int(
                    checkpoint.model_checkpoint_path.split("/")[-1].split("-")[-1]
                )
                model.load(sess, checkpoint.model_checkpoint_path)
        else:
            start_step = 0

        # Print shape of all trainable variables
        trainable_vars = tf.trainable_variables()
        for var, val in zip(trainable_vars, sess.run(trainable_vars)):
            logging.debug("%s %s", var.name, var.get_shape())

        logging.debug("starting training")

        for update_step in range(start_step, num_steps):
            trainer.step(sess, update_step)

            if (update_step % log_interval == 0) or (update_step + 1) == num_steps:
                test_start_time = timeit.default_timer()

                # Evaluate training set
                train_metrics = trainer.evaluate_metrics(
                    sess, train_metrics_handler, prefix="train", decimation=10000
                )

                # Evaluate validation set
                if validation_handler:
                    val_metrics = trainer.evaluate_metrics(sess, validation_handler, prefix="val", decimation=1000)
                else:
                    val_metrics = {}

                all_metrics = {**train_metrics, **val_metrics}
                metric_string = " ".join(
                    ["%s=%f" % (key, val) for key, val in all_metrics.items()]
                )

                end_time = timeit.default_timer()
                test_end_time = timeit.default_timer()
                logging.info(
                    "t=%.1f (%.1f) %d %s lr=%f",
                    end_time - start_time,
                    test_end_time - test_start_time,
                    update_step,
                    metric_string,
                    trainer.get_learning_rate(update_step),
                )
                start_time = timeit.default_timer()

                # Do early stopping using validation data (if available)
                if validation_handler:
                    if all_metrics["val_mae"] < best_val_mae:
                        model.save(
                            sess, logs_path + "model.ckpt", global_step=update_step
                        )
                        best_val_mae = all_metrics["val_mae"]
                        best_val_step = update_step
                        logging.info(
                            "best_val_mae=%f, best_val_step=%d",
                            best_val_mae,
                            best_val_step,
                        )
                    if (update_step - best_val_step) > 100e6:
                        logging.info(
                            "best_val_mae=%f, best_val_step=%d",
                            best_val_mae,
                            best_val_step,
                        )
                        logging.info("No improvement in last 1e6 steps, stopping...")
                        model.save(
                            sess, logs_path + "model.ckpt", global_step=update_step
                        )
                        return
                else:
                    model.save(sess, logs_path + "model.ckpt", global_step=update_step)

def main():
    args = get_arguments()
    try:
        basename = os.path.basename(args.dataset)
    except:
        basename = "."

    logs_path = "logs/%s/" % (basename.split(".")[0])
    os.makedirs(logs_path, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(logs_path + "printlog.txt", mode="w"),
            logging.StreamHandler(),
        ],
    )
    logging.debug("logging to %s" % logs_path)

    if not args.load_model:
        with open(logs_path + "commandline_args.txt", "w") as f:
            f.write("\n".join(sys.argv[1:]))

    train_model(args, logs_path)

if __name__ == "__main__":
    main()
