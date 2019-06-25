import sys
import argparse
import os
import timeit
import logging
import ase
import ase.io
import msgnet
import tensorflow as tf
import numpy as np
import densitymsg
from densityloader import VaspChargeDataLoader
from densityhandler import DensityDataHandler
from trainer import DensityOutputTrainer
from ase.neighborlist import NeighborList

CUTOFF_ANGSTROM = 5.0

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Train graph convolution network", fromfile_prefix_chars="@"
    )
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--plot_density", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    return parser.parse_args(arg_list)

class ReadoutLastnode(msgnet.readout.ReadoutFunction):
    is_sum = False

    def __call__(self, nodes, segments):
        nodes_size = int(nodes.get_shape()[1])
        set_len = tf.segment_sum(tf.ones_like(segments), segments, name="set_len")
        last_node_idx = tf.cumsum(set_len) - 1
        last_nodes = tf.gather(nodes, last_node_idx)
        graph_out = msgnet.defaults.mlp(
            last_nodes,
            [nodes_size, nodes_size, self.output_size],
            activation=msgnet.defaults.nonlinearity,
            weights_initializer=msgnet.defaults.initializer,
        )
        return graph_out

def get_model():
    embedding_size = 128

    model = densitymsg.DensityMsgPassing(
        embedding_shape=(len(ase.data.chemical_symbols), embedding_size),
        edge_feature_expand=[(0, 0.01, CUTOFF_ANGSTROM+1)],
        use_edge_updates=True,
        hard_cutoff=CUTOFF_ANGSTROM)

    return model

def train_model(args, logs_path):
    densityloader = VaspChargeDataLoader("vaspchgcar.tar.gz", CUTOFF_ANGSTROM)
    graph_obj_list = densityloader.load()

    data_handler = DensityDataHandler(graph_obj_list)
    train_handler, _, validation_handler = data_handler.train_test_split(split_type="count", validation_size=3, test_size=0)

    batch_size = 1

    model = get_model()

    trainer = DensityOutputTrainer(model, train_handler, batch_size=batch_size, initial_lr=args.learning_rate)

    num_steps = int(1e6)
    start_step = 0
    log_interval = 1000

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
                    sess, train_handler, prefix="train", decimation=1000
                )

                # Evaluate validation set
                if validation_handler:
                    val_metrics = trainer.evaluate_metrics(sess, validation_handler, prefix="val")
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
                    if (update_step - best_val_step) > 1e6:
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

def plot_prediction(model_file):
    from mayavi import mlab
    model = get_model()
    densityloader = VaspChargeDataLoader("vaspchgcar.zip", CUTOFF_ANGSTROM)
    graph_obj_list = densityloader.load()

    data_handler = msgnet.datahandler.EdgeSelectDataHandler(graph_obj_list, ["density"], [0])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        model.load(sess, model_file)

        density = []
        target_density = []
        for input_data in data_handler.get_test_batches(10):
            feed_dict = {}
            for key, val in model.get_input_symbols().items():
                feed_dict[val] = input_data[key]
            test_pred, = sess.run([model.get_graph_out()], feed_dict=feed_dict)
            density.append(test_pred)
            target_density.append(input_data["graph_targets"])

        pred_density = np.concatenate(density)
        target_density = np.concatenate(target_density)

    pred_density = pred_density.reshape(densityloader.grid_pos.shape[0:3])
    target_density = target_density.reshape(densityloader.grid_pos.shape[0:3])

    errors = target_density-pred_density
    rmse = np.sqrt(np.mean(np.square(errors)))
    mae = np.mean(np.abs(errors))

    print("mae=%f, rmse=%f" % (mae, rmse))

    x = densityloader.grid_pos[:,:,:,0]
    y = densityloader.grid_pos[:,:,:,1]
    z = densityloader.grid_pos[:,:,:,2]

    mlab.contour3d(x,y,z,pred_density)
    mlab.contour3d(x,y,z,target_density)
    mlab.contour3d(x,y,z,errors)

    mlab.show()

def main(_):
    logs_path = "logs/"
    os.makedirs(logs_path, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(logs_path + "printlog.txt", mode="w"),
            logging.StreamHandler(),
        ],
    )
    logging.debug("ping")
    args = get_arguments()
    if args.plot_density:
        plot_prediction(args.plot_density)
    else:
        train_model(args, logs_path)

if __name__ == "__main__":
    tf.app.run()
