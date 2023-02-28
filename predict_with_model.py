import logging
import os
import json
import argparse
import math
import contextlib
import timeit

import ase
import ase.io
import torch
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
    parser.add_argument("--vacuum", type=float, default=1.0, help="Pad simulation box with vacuum (only used when boundary conditions are not periodic)")
    parser.add_argument("--output_dir", type=str, default="model_prediction", help="Output directory")
    parser.add_argument("--iri", action="store_true", help="Also compute interaction region indicator (IRI)")
    parser.add_argument("--dori", action="store_true", help="Also compute density overlap region indicator (DORI)")
    parser.add_argument("--hessian_eig", action="store_true", help="Also compute eigenvalues of density Hessian")
    parser.add_argument("--probe_count", type=int, default=5000, help="How many probe points to compute per iteration")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Set which device to use for inference e.g. 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--ignore_pbc",
        action="store_true",
        help="If flag is given, disable periodic boundary conditions (force to False) in atoms data",
    )
    parser.add_argument(
        "--force_pbc",
        action="store_true",
        help="If flag is given, force periodic boundary conditions to True in atoms data",
    )

    return parser.parse_args(arg_list)

def load_model(model_dir, device):
    with open(os.path.join(model_dir, "arguments.json"), "r") as f:
        runner_args = argparse.Namespace(**json.load(f))
    if runner_args.use_painn_model:
        model = densitymodel.PainnDensityModel(runner_args.num_interactions, runner_args.node_size, runner_args.cutoff)
    else:
        model = densitymodel.DensityModel(runner_args.num_interactions, runner_args.node_size, runner_args.cutoff)
    device = torch.device(device)
    model.to(device)
    state_dict = torch.load(os.path.join(model_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(state_dict["model"])
    return model, runner_args.cutoff

class LazyMeshGrid():
    def __init__(self, cell, grid_step, origin=None, adjust_grid_step=False):
        self.cell = cell
        if adjust_grid_step:
            n_steps = np.round(self.cell.lengths()/grid_step)
            self.scaled_grid_vectors = [np.arange(n)/n for n in n_steps]
            self.adjusted_grid_step = self.cell.lengths()/n_steps
        else:
            self.scaled_grid_vectors = [np.arange(0, l, grid_step)/l for l in self.cell.lengths()]
        self.shape = np.array([len(g) for g in self.scaled_grid_vectors] + [3])
        if origin is None:
            self.origin = np.zeros(3)
        else:
            self.origin = origin

        self.origin = np.expand_dims(self.origin, 0)

    def __getitem__(self, indices):
        indices = np.array(indices)
        indices_shape = indices.shape
        if not (len(indices_shape) == 2 and indices_shape[0] == 3):
            raise NotImplementedError("Indexing must be a 3xN array-like object")
        gridA = self.scaled_grid_vectors[0][indices[0]]
        gridB = self.scaled_grid_vectors[1][indices[1]]
        gridC = self.scaled_grid_vectors[2][indices[2]]

        grid_pos = np.stack([gridA, gridB, gridC], 1)
        grid_pos = np.dot(grid_pos, self.cell)
        grid_pos += self.origin

        return grid_pos


def ceil_float(x, step_size):
    # Round up to nearest step_size and subtract a small epsilon
    x = math.ceil(x/step_size) * step_size
    eps = 2*np.finfo(float).eps * x
    return x - eps

def load_atoms(atomspath, vacuum, grid_step):
    atoms = ase.io.read(atomspath)

    if np.any(atoms.get_pbc()):
        atoms, grid_pos, origin = load_material(atoms, grid_step)
    else:
        atoms, grid_pos, origin = load_molecule(atoms, grid_step, vacuum)

    metadata = {"filename": atomspath}
    res = {
        "atoms": atoms,
        "origin": origin,
        "grid_position": grid_pos,
        "metadat": metadata,
    }

    return res

def load_material(atoms, grid_step):
    atoms = atoms.copy()
    grid_pos = LazyMeshGrid(atoms.get_cell(), grid_step, adjust_grid_step=True)
    origin = np.zeros(3)

    return atoms, grid_pos, origin

def load_molecule(atoms, grid_step, vacuum):
    atoms = atoms.copy()
    atoms.center(vacuum=vacuum) # This will create a cell around the atoms

    # Readjust cell lengths to be a multiple of grid_step
    a, b, c, ang_bc, ang_ac, ang_ab = atoms.get_cell_lengths_and_angles()
    a, b, c = ceil_float(a, grid_step), ceil_float(b, grid_step), ceil_float(c, grid_step)
    atoms.set_cell([a, b, c, ang_bc, ang_ac, ang_ab])

    origin = np.zeros(3)

    grid_pos = LazyMeshGrid(atoms.get_cell(), grid_step)

    return atoms, grid_pos, origin
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
        density_dict["grid_position"].shape[0:3],
        density_dict["origin"],
        "predicted by DeepDFT model",
    )
    if args.iri:
        cubewriter_iri = utils.CubeWriter(
            os.path.join(args.output_dir, "iri.cube"),
            density_dict["atoms"],
            density_dict["grid_position"].shape[0:3],
            density_dict["origin"],
            "predicted by DeepDFT model",
        )
    if args.dori:
        cubewriter_dori = utils.CubeWriter(
            os.path.join(args.output_dir, "dori.cube"),
            density_dict["atoms"],
            density_dict["grid_position"].shape[0:3],
            density_dict["origin"],
            "predicted by DeepDFT model",
        )
    if args.hessian_eig:
        cubewriter_hessian_eig = []
        for i in range(3):
            cubewriter_hessian_eig.append(
                utils.CubeWriter(
                    os.path.join(args.output_dir, "hessian_eig_%d.cube" % i),
                    density_dict["atoms"],
                    density_dict["grid_position"].shape[0:3],
                    density_dict["origin"],
                    "predicted by DeepDFT model",
                )
            )

    if args.ignore_pbc and args.force_pbc:
        raise ValueError("ignore_pbc and force_pbc are mutually exclusive and can't both be set at the same time")
    elif args.ignore_pbc:
        set_pbc = False
    elif args.force_pbc:
        set_pbc = True
    else:
        set_pbc = None

    start_time = timeit.default_timer()

    if args.iri or args.dori or args.hessian_eig:
        contextmanager = contextlib.nullcontext()
    else:
        # No gradients needed from the model
        contextmanager = torch.no_grad()
    with contextmanager:
        # Make graph with no probes
        logging.debug("Computing atom-to-atom graph")
        collate_fn = dataset.CollateFuncAtoms(
            cutoff=cutoff,
            pin_memory=device.type == "cuda",
            set_pbc_to=set_pbc,
        )
        graph_dict = collate_fn([density_dict])
        logging.debug("Computing atom representation")
        device_batch = {
            k: v.to(device=device, non_blocking=True) for k, v in graph_dict.items()
        }
        if isinstance(model, densitymodel.PainnDensityModel):
            atom_representation_scalar, atom_representation_vector = model.atom_model(device_batch)
        else:
            atom_representation = model.atom_model(device_batch)
        logging.debug("Atom representation done")

        # Loop over all slices
        density_iter = dataset.DensityGridIterator(density_dict, args.probe_count, cutoff, set_pbc_to=set_pbc)
        density = []
        for probe_graph_dict in density_iter:
            probe_dict = dataset.collate_list_of_dicts([probe_graph_dict])
            probe_dict = {
                k: v.to(device=device, non_blocking=True) for k, v in probe_dict.items()
            }
            device_batch["probe_edges"] = probe_dict["probe_edges"]
            device_batch["probe_edges_displacement"] = probe_dict["probe_edges_displacement"]
            device_batch["probe_xyz"] = probe_dict["probe_xyz"]
            device_batch["num_probe_edges"] = probe_dict["num_probe_edges"]
            device_batch["num_probes"] = probe_dict["num_probes"]

            if isinstance(model, densitymodel.PainnDensityModel):
                res = model.probe_model(device_batch, atom_representation_scalar, atom_representation_vector, compute_iri=args.iri, compute_dori=args.dori, compute_hessian=args.hessian_eig)
            else:
                res = model.probe_model(device_batch, atom_representation, compute_iri=args.iri, compute_dori=args.dori, compute_hessian=args.hessian_eig)

            if args.iri or args.dori or args.hessian_eig:
                density, grad_outputs = res
            else:
                density = res
            if args.iri:
                iri = grad_outputs["iri"].cpu().detach().numpy().flatten()
                cubewriter_iri.write(iri)
            if args.dori:
                cubewriter_dori.write(grad_outputs["dori"].cpu().detach().numpy().flatten())
            if args.hessian_eig:
                eigs = torch.linalg.eigvalsh(grad_outputs["hessian"])
                eiglist = torch.unbind(eigs, dim=-1)
                for writer, val in zip(cubewriter_hessian_eig, eiglist):
                    writer.write(val.cpu().detach().numpy().flatten())

            cubewriter.write(density.cpu().detach().numpy().flatten())
            logging.debug("Written %d/%d", cubewriter.numbers_written, np.prod(density_dict["grid_position"].shape[0:3]))

    end_time = timeit.default_timer()

    logging.info("done time_elapsed=%f", end_time-start_time)

if __name__ == "__main__":
    main()
