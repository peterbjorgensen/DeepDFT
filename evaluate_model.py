import numpy as np
import tensorflow as tf
from densityloader import ChargeDataLoader
from densityhandler import DensityDataHandler
from runner import get_model, get_arguments, CUTOFF_ANGSTROM

def main():
    model = get_model()
    args = get_arguments()
    densityloader = ChargeDataLoader(args.dataset, CUTOFF_ANGSTROM)
    graph_obj_list = densityloader.load()
    data_handler = DensityDataHandler(graph_obj_list)

    train_handler, test_handler, validation_handler = data_handler.train_test_split(split_type="count", validation_size=10, test_size=0)
    data_splits = {"train": train_handler, "test": test_handler, "validation": validation_handler}

    for key,datahandler in data_splits.items():
        for gobj in datahandler.graph_objects:
            data_handler = DensityDataHandler([gobj])

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                model.load(sess, args.load_model)

                density = []
                target_density = []
                for input_data in data_handler.get_test_batches(100):
                    feed_dict = {}
                    for key, val in model.get_input_symbols().items():
                        feed_dict[val] = input_data[key]
                    test_pred, = sess.run([model.get_graph_out()], feed_dict=feed_dict)
                    density.append(test_pred.squeeze(0))
                    target_density.append(input_data["probes_target"])

                pred_density = np.concatenate(density)
                target_density = np.concatenate(target_density)

            pred_density = pred_density.reshape(data_handler.graph_objects[0].grid_position.shape[0:3])
            target_density = target_density.reshape(data_handler.graph_objects[0].grid_position.shape[0:3])

            errors = target_density-pred_density
            rmse = np.sqrt(np.mean(np.square(errors)))
            mae = np.mean(np.abs(errors))

            print("split=%s, filename=%s, mae=%f, rmse=%f" % (key, gobj.filename, mae, rmse))


if __name__ == "__main__":
    main()
