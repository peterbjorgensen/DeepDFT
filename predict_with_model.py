import ase
import ase.io
import torch
import logging
import os
import json
import argparse

import numpy as np

import dataset
import densitymodel
import utils

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Predict with pretrained model", fromfile_prefix_chars="@"
    )
    parser.add_argument("model_dir", type=str, help='Directory of pretrained model')
    parser.add_argument("atoms_file", type=str, help='ASE compatible atoms xyz-file')
    parser.add_argument("--grid_step", type=float, default=0.05, help="Step size in Ångstrøm")
    parser.add_argument("--vacuum", type=float, default=1.0, help="Pad simulation box with vacuum")
    parser.add_argument("--output_dir", type=str, default="model_prediction", help="Output directory")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Set which device to use for inference e.g. 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--ignore_pbc",
        action="store_true",
        help="If flag is given, ignore periodic boundary conditions in atoms data",
    )

    return parser.parse_args(arg_list)

def load_model(model_dir, device):
    with open(os.path.join(model_dir, "arguments.json"), "r") as f:
        runner_args = argparse.Namespace(**json.load(f))
    model = densitymodel.DensityModel(runner_args.num_interactions, runner_args.node_size, runner_args.cutoff)
    device = torch.device(device)
    model.to(device)
    state_dict = torch.load(os.path.join(model_dir, "best_model.pth"))
    model.load_state_dict(state_dict["model"])
    return model, runner_args.cutoff

class LazyMeshGrid():
    def __init__(self, cell, grid_step):
        self.cell = cell
        self.shape = np.floor(self.cell.lengths()/grid_step).astype(int)
        self.scaled_grid_vectors = [np.linspace(0, 1, a) for a in self.shape]
        #self.grid_vectors = [np.arange(a)*grid_step for a in self.shape]

    def __getitem__(self, indices):
        gridA = self.scaled_grid_vectors[0][indices[0]]
        gridB = self.scaled_grid_vectors[1][indices[1]]
        gridC = self.scaled_grid_vectors[2][indices[2]]

        grid_pos = np.stack([gridA, gridB, gridC], 1)
        grid_pos = np.dot(grid_pos, self.cell)

        return grid_pos


def load_atoms(atomspath, vacuum, grid_step):
    atoms = ase.io.read(atomspath)
    atoms.center(vacuum=vacuum)

    origin = np.zeros(3)

    grid_pos = LazyMeshGrid(atoms.get_cell(), grid_step)

    metadata = {"filename": atomspath}
    return {
        # "density": density,
        "atoms": atoms,
        "origin": origin,
        "grid_position": grid_pos,
        "metadata": metadata, # Meta information
    }

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

    model, cutoff = load_model(args.model_dir, args.device)

    density_dict = load_atoms(args.atoms_file, args.vacuum, args.grid_step)

    device = torch.device(args.device)

    cubewriter = utils.CubeWriter(
        os.path.join(args.output_dir, "prediction.cube"),
        density_dict["atoms"],
        density_dict["grid_position"].shape,
        density_dict["origin"],
        "predicted by DeepDFT model",
    )

    with torch.no_grad():
        # Make graph with no probes
        logging.debug("Computing atom-to-atom graph")
        collate_fn = dataset.CollateFuncAtoms(
            cutoff=cutoff,
            pin_memory=True,
            disable_pbc=args.ignore_pbc,
        )
        graph_dict = collate_fn([density_dict])
        logging.debug("Computing atom representation")
        device_batch = {
            k: v.to(device=device, non_blocking=True) for k, v in graph_dict.items()
        }
        atom_representation = model.atom_model(device_batch)
        logging.debug("Atom representation done")

        # Loop over all slices
        density_iter = dataset.DensityGridIterator(density_dict, args.ignore_pbc, 1000, cutoff)
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

            cubewriter.write(model.probe_model(device_batch, atom_representation).cpu().detach().numpy().flatten())
            logging.debug("Written %d/%d", cubewriter.numbers_written, np.prod(density_dict["grid_position"].shape))

if __name__ == "__main__":
    main()
