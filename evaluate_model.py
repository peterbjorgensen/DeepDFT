import zlib
import os
import tarfile
import argparse
import io
import tensorflow as tf
import numpy as np
import ase
from densityloader import ChargeDataLoader
from densityhandler import DensityDataHandler
from runner import get_model, CUTOFF_ANGSTROM

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Evaluate density model", fromfile_prefix_chars="@"
    )
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output_name", type=str, default=None)

    return parser.parse_args(arg_list)

def write_cube_to_tar(tar, atoms, cubedata, filename):
    cbuf = io.BytesIO()
    ase.io.cube.write_cube(
        cbuf,
        atoms,
        data=cubedata,
        origin=(0,0,0),
        comment=filename,
    )
    cbuf.seek(0)
    cbytes = zlib.compress(cbuf)
    fsize = len(cbytes)
    cbuf = io.BytesIO(cbytes)
    cbuf.seek(0)
    tarinfo = tarfile.TarInfo(name=filename)
    tarinfo.size = fsize
    tar.addfile(tarinfo, cbuf)

def main():
    model = get_model()
    args = get_arguments()
    densityloader = ChargeDataLoader(args.dataset, CUTOFF_ANGSTROM)
    graph_obj_list = densityloader.load()
    data_handler = DensityDataHandler(graph_obj_list)

    train_handler, test_handler, validation_handler = data_handler.train_test_split(split_type="count", validation_size=10, test_size=0)
    data_splits = {"train": train_handler, "test": test_handler, "validation": validation_handler}

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        model.load(sess, args.load_model)

        logfile = open("evaluate_model.txt", "w")

        for split_name, datahandler in data_splits.items():
            if args.output_name:
                head, tail = os.path.split(args.output_name)
                outname = os.path.join(head, tail+"_"+split_name+".tar")
                tar = tarfile.open(outname, "w")

            for cobj in datahandler.graph_objects:
                gobj = cobj.decompress()
                data_handler = DensityDataHandler([gobj])

                density = []
                target_density = []
                for input_data in data_handler.get_test_batches(100):
                    feed_dict = {}
                    for key, val in model.get_input_symbols().items():
                        feed_dict[val] = input_data[key]
                    test_pred, = sess.run([model.get_graph_out()], feed_dict=feed_dict)
                    density.append(test_pred.squeeze(0))
                    target_density.append(input_data["probes_target"].squeeze(0))

                    pred_density = np.concatenate(density)
                    target_density = np.concatenate(target_density)

                pred_density = pred_density.reshape(data_handler.graph_objects[0].grid_position.shape[0:3])
                target_density = target_density.reshape(data_handler.graph_objects[0].grid_position.shape[0:3])

                errors = target_density-pred_density
                rmse = np.sqrt(np.mean(np.square(errors)))
                mae = np.mean(np.abs(errors))

                if args.output_name is not None:
                    write_cube_to_tar(
                        tar,
                        data_handler.graph_objects[0].atoms,
                        pred_density,
                        data_handler.graph_objects[0].filename,
                        )

                print("split=%s, filename=%s, mae=%f, rmse=%f" % (split_name, gobj.filename, mae, rmse))
                print("split=%s, filename=%s, mae=%f, rmse=%f" % (split_name, gobj.filename, mae, rmse), file=logfile)


if __name__ == "__main__":
    main()
