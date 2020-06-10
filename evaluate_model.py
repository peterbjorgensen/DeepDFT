import argparse
import zlib
import os
import tarfile
import io
import time
import logging

import ase
import ase.units
import numpy as np
import torch

import dataset
import densitymodel
from runner import split_data

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Evaluate density model", fromfile_prefix_chars="@"
    )
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--num_interactions", type=int, default=3)
    parser.add_argument("--node_size", type=int, default=64)
    parser.add_argument("--split_file", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Set which device to use for training e.g. 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--ignore_pbc",
        action="store_true",
        help="If flag is given, ignore periodic boundary conditions in atoms data",
    )

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
    """write_cube_to_tar
    Write cube file to tar archive and compress the file using zlib.
    Cubedata is expected to be in electrons/A^3 and is converted to
    electrons/Bohr^3, which is cube file convention

    :param tar:
    :param atoms:
    :param cubedata:
    :param origin:
    :param filename:
    """
    cbuf = io.StringIO()
    write_cube(
        cbuf,
        atoms,
        data=cubedata*(ase.units.Bohr**3),
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
    args = get_arguments()

    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(args.output_dir, "printlog.txt"), mode="w"
            ),
            logging.StreamHandler(),
        ],
    )

    # Initialise model and load model
    device = torch.device(args.device)
    net = densitymodel.DensityModel(args.num_interactions, args.node_size, args.cutoff,)
    net = net.to(device)
    state_dict = torch.load(args.load_model)
    net.load_state_dict(state_dict["model"])

    # Load dataset
    if args.dataset.endswith(".txt"):
        # Text file contains list of datafiles
        with open(args.dataset, "r") as datasetfiles:
            filelist = [os.path.join(os.path.dirname(args.dataset), line.strip('\n')) for line in datasetfiles]
    else:
        filelist = [args.dataset]
    logging.info("loading data %s", args.dataset)
    densitydata = torch.utils.data.ConcatDataset([dataset.DensityData(path) for path in filelist])

    # Split data into train and validation sets
    datasplits = split_data(densitydata, args)


    for split_name, densitydataset in datasplits.items():
        if args.output_dir:
            outname = os.path.join(args.output_dir, "eval_" + split_name + ".tar")
            tar = tarfile.open(outname, "w")

        for density_dict in densitydataset:
            with torch.no_grad():
                # Make graph with no probes
                collate_fn = dataset.CollateFuncRandomSample(
                    cutoff=args.cutoff,
                    num_probes=0,
                    pin_memory=True,
                    disable_pbc=args.ignore_pbc,
                )
                graph_dict = collate_fn([density_dict])
                device_batch = {
                    k: v.to(device=device, non_blocking=True) for k, v in graph_dict.items()
                }
                atom_representation = net.atom_model(device_batch)

                # Loop over all slices
                density_iter = dataset.DensityGridIterator(density_dict, args.ignore_pbc, 1000, args.cutoff)
                density = []
                for probe_graph_dict in density_iter:
                    probe_dict = dataset.collate_list_of_dicts([probe_graph_dict])
                    probe_dict = {
                        k: v.to(device=device, non_blocking=True) for k, v in probe_dict.items()
                    }
                    device_batch["probe_edges"] = probe_dict["probe_edges"]
                    device_batch["probe_edges_features"] = probe_dict["probe_edges_features"]
                    device_batch["num_probe_edges"] = probe_dict["num_probe_edges"]
                    device_batch["num_probes"] = probe_dict["num_probes"]

                    density.append(net.probe_model(device_batch, atom_representation).cpu().detach().numpy())

            pred_density = np.concatenate(density, axis=1)
            target_density = density_dict["density"]
            pred_density = pred_density.reshape(target_density.shape)

            errors = target_density-pred_density
            rmse = np.sqrt(np.mean(np.square(errors)))
            mae = np.mean(np.abs(errors))

            if args.output_dir is not None:
                fname_stripped = density_dict["metadata"]["filename"]
                while fname_stripped.endswith(".zz"):
                    fname_stripped = fname_stripped[:-3]
                name, _ = os.path.splitext(fname_stripped)
                write_cube_to_tar(
                    tar,
                    density_dict["atoms"],
                    pred_density,
                    density_dict["grid_position"][0, 0, 0],
                    name + "_prediction" + ".cube" + ".zz",
                    )
                write_cube_to_tar(
                    tar,
                    density_dict["atoms"],
                    errors,
                    density_dict["grid_position"][0, 0, 0],
                    name + "_error" + ".cube" + ".zz",
                    )
                write_cube_to_tar(
                    tar,
                    density_dict["atoms"],
                    target_density,
                    density_dict["grid_position"][0, 0, 0],
                    name + "_target" + ".cube" + ".zz",
                    )

            logging.info("split=%s, filename=%s, mae=%f, rmse=%f", split_name, density_dict["metadata"]["filename"], mae, rmse)


if __name__ == "__main__":
    main()
