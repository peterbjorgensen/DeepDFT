import zlib
import os
import tarfile
import argparse
import io
import tensorflow as tf
import numpy as np
import ase
import ase.units
import time
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

def write_cube(fileobj, atoms, data=None, origin=None, comment=None):
    """
    Function to write a cube file. This is a copy of ase.io.cube.write_cube but supports
    textIO buffer

    fileobj: str or file object
        File to which output is written.
    atoms: Atoms object
        Atoms object specifying the atomic configuration.
    data : 3dim numpy array, optional (default = None)
        Array containing volumetric data as e.g. electronic density
    origin : 3-tuple
        Origin of the volumetric data (units: Angstrom)
    comment : str, optional (default = None)
        Comment for the first line of the cube file.
    """

    if data is None:
        data = np.ones((2, 2, 2))
    data = np.asarray(data)

    if data.dtype == complex:
        data = np.abs(data)

    if comment is None:
        comment = 'Cube file from ASE, written on ' + time.strftime('%c')
    else:
        comment = comment.strip()
    fileobj.write(comment)

    fileobj.write('\nOUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n')

    if origin is None:
        origin = np.zeros(3)
    else:
        origin = np.asarray(origin) / ase.units.Bohr

    fileobj.write('{0:5}{1:12.6f}{2:12.6f}{3:12.6f}\n'
                  .format(len(atoms), *origin))

    for i in range(3):
        n = data.shape[i]
        d = atoms.cell[i] / n / ase.units.Bohr
        fileobj.write('{0:5}{1:12.6f}{2:12.6f}{3:12.6f}\n'.format(n, *d))

    positions = atoms.positions / ase.units.Bohr
    numbers = atoms.numbers
    for Z, (x, y, z) in zip(numbers, positions):
        fileobj.write('{0:5}{1:12.6f}{2:12.6f}{3:12.6f}{4:12.6f}\n'
                      .format(Z, 0.0, x, y, z))

    for el in data.flat:
        fileobj.write("%e\n" % el)


def write_cube_to_tar(tar, atoms, cubedata, origin, filename):
    cbuf = io.StringIO()
    write_cube(
        cbuf,
        atoms,
        data=cubedata,
        origin=origin,
        comment=filename,
    )
    cbuf.seek(0)
    cube_bytes = cbuf.getvalue().encode()
    cbytes = zlib.compress(cube_bytes)
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

    train_handler, test_handler, validation_handler = data_handler.train_test_split(split_type="count", validation_size=10, test_size=10)
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
                    name, ext = os.path.splitext(data_handler.graph_objects[0].filename)
                    write_cube_to_tar(
                        tar,
                        data_handler.graph_objects[0].atoms,
                        pred_density,
                        data_handler.graph_objects[0].grid_position[0,0,0],
                        name + "_prediction" + ext + ".zz",
                        )
                    write_cube_to_tar(
                        tar,
                        data_handler.graph_objects[0].atoms,
                        errors,
                        data_handler.graph_objects[0].grid_position[0,0,0],
                        name + "_error" + ext + ".zz",
                        )
                    write_cube_to_tar(
                        tar,
                        data_handler.graph_objects[0].atoms,
                        target_density,
                        data_handler.graph_objects[0].grid_position[0,0,0],
                        name + "_target" + ext + ".zz",
                        )

                print("split=%s, filename=%s, mae=%f, rmse=%f" % (split_name, gobj.filename, mae, rmse))
                print("split=%s, filename=%s, mae=%f, rmse=%f" % (split_name, gobj.filename, mae, rmse), file=logfile)


if __name__ == "__main__":
    main()
