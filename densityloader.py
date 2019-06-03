import numpy as np
import multiprocessing
import itertools
import msgnet
import ase
import warnings
from msgnet.dataloader import FeatureGraph
from ase.neighborlist import NeighborList
from ase.calculators.vasp import VaspChargeDensity

class FeatureGraphVirtual(FeatureGraph):
    @staticmethod
    def atoms_to_graph_const_cutoff(
        atoms: ase.Atoms,
        cutoff,
        atom_to_node_fn,
        self_interaction=False,
        cutoff_covalent=False,
    ):

        atoms.wrap()
        atom_numbers = atoms.get_atomic_numbers()

        if cutoff_covalent:
            radii = ase.data.covalent_radii[atom_numbers] * cutoff
        else:
            radii = [cutoff] * len(atoms)
        neighborhood = NeighborList(
            radii, skin=0.0, self_interaction=self_interaction, bothways=True, primitive=ase.neighborlist.NewPrimitiveNeighborList
        )
        neighborhood.update(atoms)

        nodes = []
        connections = []
        connections_offset = []
        edges = []
        if np.any(atoms.get_pbc()):
            atom_positions = atoms.get_positions(wrap=True)
        else:
            atom_positions = atoms.get_positions(wrap=False)
        unitcell = atoms.get_cell()

        for ii in range(len(atoms)):
            nodes.append(atom_to_node_fn(atom_numbers[ii]))

        for ii in range(len(atoms)):
            neighbor_indices, offset = neighborhood.get_neighbors(ii)
            for jj, offs in zip(neighbor_indices, offset):
                if atom_numbers[jj] == 0:
                    continue # The probe atom (number 0) has no outgoing connections
                ii_pos = atom_positions[ii]
                jj_pos = atom_positions[jj] + np.dot(offs, unitcell)
                dist_vec = ii_pos - jj_pos
                dist = np.sqrt(np.dot(dist_vec, dist_vec))

                connections.append([jj, ii])
                connections_offset.append([[offs[0], offs[1], offs[2]], [0,0,0]])
                edges.append([dist])

        if len(edges) == 0:
            warnings.warn("Generated graph has zero edges")
            edges = np.zeros((0, 1))
            connections = np.zeros((0, 2))
            connections_offset = np.zeros((0, 2, 3))

        return (
            np.array(nodes),
            atom_positions,
            np.array(edges),
            np.array(connections),
            np.array(connections_offset),
            unitcell,
        )

class VaspChargeDataLoader(msgnet.dataloader.DataLoader):
    def __init__(self, vasp_fname, cutoff_radius, subsample_factor=1, prefix=""):
        super().__init__()
        vasp_charge = VaspChargeDensity(filename=vasp_fname)
        self.download_dest = vasp_fname
        self.vasp_charge = vasp_charge
        self.subsample_factor = subsample_factor
        self.prefix = prefix
        self.density = vasp_charge.chg[-1][::subsample_factor, ::subsample_factor, ::subsample_factor] #seperate density
        self.atoms = vasp_charge.atoms[-1] #seperate atom positions
        ngridpts = np.array(self.density.shape) #grid matrix
        self.cutoff_radius = cutoff_radius

        grid_pos = np.meshgrid(
            np.arange(ngridpts[0])/self.density.shape[0],
            np.arange(ngridpts[1])/self.density.shape[1],
            np.arange(ngridpts[2])/self.density.shape[2],
            indexing='ij',
        )
        grid_pos = np.stack(grid_pos, 3)
        self.grid_pos = np.dot(grid_pos, self.atoms.get_cell())

    @property
    def final_dest(self):
        cutname = "%s-%.2f" % (self.cutoff_type, self.cutoff_radius)
        if self.subsample_factor > 1:
            cutname += "-ss%d" % self.subsample_factor
        return "/%s/%s%s_%s.pkz" % (
            msgnet.defaults.datadir,
            self.prefix,
            self.__class__.__name__,
            cutname,
        )

    def _download_data(self):
        pass

    def _preprocess(self):
        num_pos = np.prod(self.grid_pos.shape[0:3])
        probe_pos = []
        target_density = []
        for i in range(num_pos):
            grid_index = np.unravel_index(i, self.grid_pos.shape[0:3])
            probe_pos.append(self.grid_pos[tuple(grid_index)])
            target_density.append(self.density[tuple(grid_index)])

        pool = multiprocessing.Pool(4)
        input_params = zip(
            itertools.repeat(self.atoms, len(probe_pos)),
            probe_pos,
            target_density,
            itertools.repeat(self.cutoff_radius, len(probe_pos)))
        for i, res in enumerate(pool.imap(preprocess_worker, input_params)):
            if i % 100 == 0:
                print("%010d    " % i, sep="", end="\r")
            yield res
        pool.close()
        print("")

def preprocess_worker(input_tuple):
    atom, probe_pos, target_density, cutoff_radius = input_tuple
    probe_atom = ase.atom.Atom(0, probe_pos)
    atom.append(probe_atom)
    graphobj = FeatureGraphVirtual(atom, "const", cutoff_radius, lambda x: x, density=target_density)
    return graphobj
