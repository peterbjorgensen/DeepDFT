import sys
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
from ase.neighborlist import NeighborList

CUTOFF_ANGSTROM = 5.0

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
    embedding_size = 64

    model = densitymsg.DensityMsgPassing(
        embedding_shape=(len(ase.data.chemical_symbols), embedding_size),
        edge_feature_expand=[(0, 0.01, CUTOFF_ANGSTROM+1)],
        use_edge_updates=False,
        readout_fn=ReadoutLastnode())

    return model

def main():
    densityloader = VaspChargeDataLoader("si30/CHGCAR", CUTOFF_ANGSTROM, 5)
    graph_obj_list = densityloader.load()[0:20]

    data_handler = msgnet.datahandler.EdgeSelectDataHandler(graph_obj_list, ["density"], [0])

    batch_size = 20

    model = get_model()

    trainer = msgnet.train.GraphOutputTrainer(model, data_handler, batch_size=batch_size)

    num_steps = int(1e6)
    start_step = 0
    log_interval = 1000
    val_obj = None
    train_obj = data_handler

    start_time = timeit.default_timer()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Print shape of all trainable variables
        trainable_vars = tf.trainable_variables()
        for var, val in zip(trainable_vars, sess.run(trainable_vars)):
            logging.debug("%s %s", var.name, var.get_shape())

        for update_step in range(start_step, num_steps):
            trainer.step(sess, update_step)

            if (update_step % log_interval == 0) or (update_step + 1) == num_steps:
                test_start_time = timeit.default_timer()

                # Evaluate training set
                train_metrics = trainer.evaluate_metrics(
                    sess, train_obj, prefix="train"
                )

                # Evaluate validation set
                if val_obj:
                    val_metrics = trainer.evaluate_metrics(sess, val_obj, prefix="val")
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
                if val_obj:
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
    densityloader = VaspChargeDataLoader("si30/CHGCAR", CUTOFF_ANGSTROM, 5)
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

    x = densityloader.grid_pos[:,:,:,0]
    y = densityloader.grid_pos[:,:,:,1]
    z = densityloader.grid_pos[:,:,:,2]

    mlab.contour3d(x,y,z,pred_density)
    mlab.contour3d(x,y,z,target_density)

    mlab.show()


if __name__ == "__main__":
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
    if len(sys.argv) > 1:
        plot_prediction(sys.argv[1])
    else:
        main()
