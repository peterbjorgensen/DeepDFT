from typing import List
import tarfile
import multiprocessing
import threading
import logging
import zlib
import os
import io
import math
import torch
import lz4.frame
import numpy as np
import ase
import ase.io.cube
from ase.calculators.vasp import VaspChargeDensity
import asap3

from layer import pad_and_stack


def rotating_pool_worker(dataset, rng, queue):
    while True:
        for index in rng.permutation(len(dataset)):
            queue.put(dataset[index])


def transfer_thread(queue: multiprocessing.Queue, datalist: list):
    while True:
        for index in range(len(datalist)):
            datalist[index] = queue.get()


class RotatingPoolData(torch.utils.data.Dataset):
    """
    Wrapper for a dataset that continously loads data into a smaller pool.
    The data loading is performed in a separate process and is assumed to be IO bound.
    """

    def __init__(self, dataset, pool_size, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.parent_data = dataset
        self.rng = np.random.default_rng()
        logging.debug("Filling rotating data pool of size %d" % pool_size)
        self.data_pool = [
            self.parent_data[i]
            for i in self.rng.integers(
                0, high=len(self.parent_data), size=self.pool_size, endpoint=False
            )
        ]
        self.loader_queue = multiprocessing.Queue(2)

        # Start loaders
        self.loader_process = multiprocessing.Process(
            target=rotating_pool_worker,
            args=(self.parent_data, self.rng, self.loader_queue),
        )
        self.transfer_thread = threading.Thread(
            target=transfer_thread, args=(self.loader_queue, self.data_pool)
        )
        self.loader_process.start()
        self.transfer_thread.start()

    def __len__(self):
        return self.pool_size

    def __getitem__(self, index):
        return self.data_pool[index]


class BufferData(torch.utils.data.Dataset):
    """
    Wrapper for a dataset. Loads all data into memory.
    """

    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)

        self.data_objects = [dataset[i] for i in range(len(dataset))]

    def __len__(self):
        return len(self.data_objects)

    def __getitem__(self, index):
        return self.data_objects[index]


class DensityData(torch.utils.data.Dataset):
    def __init__(self, tarpath, **kwargs):
        super().__init__(**kwargs)

        self.tarpath = tarpath
        self.member_list = []

        # Index tar file
        with tarfile.open(self.tarpath, "r:") as tar:
            for member in tar.getmembers():
                self.member_list.append(member)

    def __len__(self):
        return len(self.member_list)

    def extract_member(self, tarinfo):
        with tarfile.open(self.tarpath, "r") as tar:
            if tarinfo.name.endswith(".cube"):
                density, atoms, origin = _extract_cube(tar, tarinfo)
            elif tarinfo.name.endswith(".cube.zz"):
                density, atoms, origin = _extract_compressed_cube(tar, tarinfo)
            else:
                density, atoms, origin = _extract_vasp(tar, tarinfo)

        grid_pos = _calculate_grid_pos(density, origin, atoms.get_cell())

        metadata = {"filename": tarinfo.name}
        return {
            "density": density,
            "atoms": atoms,
            "origin": origin,
            "grid_position": grid_pos,
            "metadata": metadata, # Meta information
        }

    def __getitem__(self, index):
        return self.extract_member(self.member_list[index])


class AseNeigborListWrapper:
    """
    Wrapper around ASE neighborlist to have the same interface as asap3 neighborlist

    """

    def __init__(self, cutoff, atoms):
        self.neighborlist = ase.neighborlist.NewPrimitiveNeighborList(
            cutoff, skin=0.0, self_interaction=False, bothways=True
        )
        self.neighborlist.build(
            atoms.get_pbc(), atoms.get_cell(), atoms.get_positions()
        )
        self.cutoff = cutoff
        self.atoms_positions = atoms.get_positions()
        self.atoms_cell = atoms.get_cell()

    def get_neighbors(self, i, cutoff):
        assert (
            cutoff == self.cutoff
        ), "Cutoff must be the same as used to initialise the neighborlist"

        indices, offsets = self.neighborlist.get_neighbors(i)

        rel_positions = (
            self.atoms_positions[indices]
            + offsets @ self.atoms_cell
            - self.atoms_positions[i][None]
        )

        dist2 = np.sum(np.square(rel_positions), axis=1)

        return indices, rel_positions, dist2


class DensityGridIterator:
    def __init__(self, densitydict, ignore_pbc: bool, probe_count: int, cutoff: float):
        self.densitydict = densitydict

        self.num_positions = np.prod(self.densitydict["grid_position"].shape[0:3])
        self.num_slices = int(math.ceil(self.num_positions / probe_count))
        self.probe_count = probe_count
        self.cutoff = cutoff

    def get_slice(self, slice_index):
        flat_index = np.arange(slice_index*self.probe_count, min((slice_index+1)*self.probe_count, self.num_positions))
        pos_index = np.unravel_index(flat_index, self.densitydict["grid_position"].shape[0:3])
        target_density = self.densitydict["density"][pos_index]
        probe_pos = self.densitydict["grid_position"][pos_index]
        _, _, probe_edges, probe_edges_features = atoms_and_probes_to_graph(self.densitydict["atoms"], probe_pos, self.cutoff)

        if not probe_edges:
            probe_edges = [np.zeros((0,2), dtype=np.int)]
            probe_edges_features = [np.zeros((0,), dtype=np.int)]

        # pylint: disable=E1102
        default_type = torch.get_default_dtype()
        res = {
            "probe_edges": torch.tensor(np.concatenate(probe_edges, axis=0)),
            "probe_edges_features": torch.tensor(
                np.concatenate(probe_edges_features, axis=0)[:, None], dtype=default_type
            ),
        }
        res["num_probe_edges"] = torch.tensor(res["probe_edges"].shape[0])
        res["num_probes"] = torch.tensor(len(flat_index))

        return res

    def __iter__(self):
        self.current_slice = 0
        return self

    def __next__(self):
        if self.current_slice < self.num_slices:
            this_slice = self.current_slice
            self.current_slice += 1
            return self.get_slice(this_slice)
        else:
            raise StopIteration


def atoms_and_probes_to_graph(atoms, probe_pos, cutoff):
    # Insert probe atoms
    num_probes = probe_pos.shape[0]
    probe_atoms = ase.Atoms(numbers=[0] * num_probes, positions=probe_pos)
    atoms_with_probes = atoms.copy()
    atoms_with_probes.extend(probe_atoms)

    atom_edges = []
    atom_edges_features = []
    probe_edges = []
    probe_edges_features = []

    # Compute neighborlist
    if np.any(atoms.get_cell().lengths() <= 0.0001) or (np.any(atoms.get_pbc()) and np.any(atoms.get_cell().lengths() < cutoff)):
       neighborlist = AseNeigborListWrapper(cutoff, atoms_with_probes)
    else:
        neighborlist = asap3.FullNeighborList(cutoff, atoms_with_probes)
    atomic_numbers = atoms_with_probes.get_atomic_numbers()
    for i in range(len(atoms_with_probes)):
        neigh_idx, _, neigh_dist2 = neighborlist.get_neighbors(i, cutoff)
        neigh_dist = np.sqrt(neigh_dist2)
        neigh_atomic_species = atomic_numbers[neigh_idx]

        neigh_is_atom = neigh_atomic_species != 0
        neigh_atoms = neigh_idx[neigh_is_atom]

        self_index = np.ones_like(neigh_atoms) * i
        if i < len(atoms):
            self_index = np.ones_like(neigh_atoms) * i
        else:
            self_index = np.ones_like(neigh_atoms) * (i - len(atoms))
        edges = np.stack((neigh_atoms, self_index), axis=1)

        if i < len(atoms):  # We are computing edges for an atom
            atom_edges.append(edges)
            atom_edges_features.append(neigh_dist[neigh_is_atom])
        else:  # We are computing edgs for a probe
            probe_edges.append(edges)
            probe_edges_features.append(neigh_dist[neigh_is_atom])

    return atom_edges, atom_edges_features, probe_edges, probe_edges_features

def atoms_and_probe_sample_to_graph(density, atoms, grid_pos, cutoff, num_probes):
    nodes = atoms.get_atomic_numbers()

    # Sample probes on the calculated grid
    probe_choice_max = np.prod(grid_pos.shape[0:3])
    probe_choice = np.random.randint(probe_choice_max, size=num_probes)
    probe_choice = np.unravel_index(probe_choice, grid_pos.shape[0:3])
    probe_pos = grid_pos[probe_choice]
    probe_target = density[probe_choice]

    atom_edges, atom_edges_features, probe_edges, probe_edges_features = atoms_and_probes_to_graph(atoms, probe_pos, cutoff)

    default_type = torch.get_default_dtype()

    if not probe_edges:
        probe_edges = [np.zeros((0,2), dtype=np.int)]
        probe_edges_features = [np.zeros((0,), dtype=np.int)]
    # pylint: disable=E1102
    res = {
        "nodes": torch.tensor(nodes),
        "atom_edges": torch.tensor(np.concatenate(atom_edges, axis=0)),
        "atom_edges_features": torch.tensor(
            np.concatenate(atom_edges_features, axis=0)[:, None], dtype=default_type
        ),
        "probe_edges": torch.tensor(np.concatenate(probe_edges, axis=0)),
        "probe_edges_features": torch.tensor(
            np.concatenate(probe_edges_features, axis=0)[:, None], dtype=default_type
        ),
        "probe_target": torch.tensor(probe_target, dtype=default_type),
    }
    res["num_nodes"] = torch.tensor(res["nodes"].shape[0])
    res["num_atom_edges"] = torch.tensor(res["atom_edges"].shape[0])
    res["num_probe_edges"] = torch.tensor(res["probe_edges"].shape[0])
    res["num_probes"] = torch.tensor(res["probe_target"].shape[0])

    return res

def collate_list_of_dicts(list_of_dicts, pin_memory=False):
    # Convert from "list of dicts" to "dict of lists"
    dict_of_lists = {k: [dic[k] for dic in list_of_dicts] for k in list_of_dicts[0]}

    # Convert each list of tensors to single tensor with pad and stack
    if pin_memory:
        pin = lambda x: x.pin_memory()
    else:
        pin = lambda x: x

    collated = {k: pin(pad_and_stack(dict_of_lists[k])) for k in dict_of_lists}
    return collated

class CollateFuncRandomSample:
    def __init__(self, cutoff, num_probes, pin_memory=True, disable_pbc=False):
        self.num_probes = num_probes
        self.cutoff = cutoff
        self.pin_memory = pin_memory
        self.disable_pbc = disable_pbc

    def __call__(self, input_dicts: List):
        graphs = []
        for i in input_dicts:
            if self.disable_pbc:
                atoms = i["atoms"].copy()
                atoms.set_pbc(False)
            else:
                atoms = i["atoms"]

            graphs.append(atoms_and_probe_sample_to_graph(
                i["density"],
                atoms,
                i["grid_position"],
                self.cutoff,
                self.num_probes,
            ))

        return collate_list_of_dicts(graphs, pin_memory=self.pin_memory)


def _calculate_grid_pos(density, origin, cell):
    # Calculate grid positions
    ngridpts = np.array(density.shape)  # grid matrix
    grid_pos = np.meshgrid(
        np.arange(ngridpts[0]) / density.shape[0],
        np.arange(ngridpts[1]) / density.shape[1],
        np.arange(ngridpts[2]) / density.shape[2],
        indexing="ij",
    )
    grid_pos = np.stack(grid_pos, 3)
    grid_pos = np.dot(grid_pos, cell)
    grid_pos = grid_pos + origin
    return grid_pos


def _extract_vasp(tar, tarinfo):
    # Extract compressed file
    buf = tar.extractfile(tarinfo)
    if tarinfo.name.endswith(".zz"):
        filecontent = zlib.decompress(buf.read())
    elif tarinfo.name.endswith(".lz4"):
        filecontent = lz4.frame.decompress(buf.read())
    else:
        filecontent = buf.read()

    # Write to tmp file and read using ASE
    tmppath = "/tmp/extracted%d" % os.getpid()
    with open(tmppath, "wb") as tmpfile:
        tmpfile.write(filecontent)
    vasp_charge = VaspChargeDensity(filename=tmppath)
    os.remove(tmppath)
    density = vasp_charge.chg[-1]  # separate density
    atoms = vasp_charge.atoms[-1]  # separate atom positions

    return density, atoms, np.zeros(3)  # TODO: Can we always assume origin at 0,0,0?


def _extract_cube(tar, tarinfo):
    textbuf = io.TextIOWrapper(tar.extractfile(tarinfo))
    cube = ase.io.cube.read_cube(textbuf)
    # sometimes there is an entry at index 3
    # denoting the number of values for each grid position
    origin = cube["origin"][0:3]
    # by convention the cube electron density is given in electrons/Bohr^3,
    # and ase read_cube does not convert to electrons/Å^3, so we do the conversion here
    cube["data"] *= 1.0 / ase.units.Bohr ** 3
    return cube["data"], cube["atoms"], origin


def _extract_compressed_cube(tar, tarinfo):
    buf = tar.extractfile(tarinfo)
    cube_file = io.StringIO(zlib.decompress(buf.read()).decode())
    cube = ase.io.cube.read_cube(cube_file)
    # sometimes there is an entry at index 3
    # denoting the number of values for each grid position
    origin = cube["origin"][0:3]
    # by convention the cube electron density is given in electrons/Bohr^3,
    # and ase read_cube does not convert to electrons/Å^3, so we do the conversion here
    cube["data"] *= 1.0 / ase.units.Bohr ** 3
    return cube["data"], cube["atoms"], origin
