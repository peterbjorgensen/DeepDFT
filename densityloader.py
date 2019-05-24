import numpy as np
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
            radii, skin=0.0, self_interaction=self_interaction, bothways=True
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
                connections_offset.append(np.vstack((offs, np.zeros(3, float))))
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
            np.stack(connections_offset, axis=0),
            unitcell,
        )

class VaspChargeDataLoader(msgnet.dataloader.DataLoader):
    def __init__(self, vasp_fname, cutoff_radius):
        super().__init__()
        vasp_charge = VaspChargeDensity(filename=vasp_fname)
        self.download_dest = vasp_fname
        self.vasp_charge = vasp_charge
        self.density = vasp_charge.chg[-1] #seperate density
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

    def _download_data(self):
        pass

    def _preprocess(self):
        num_pos = np.prod(self.grid_pos.shape[0:3])
        for i in range(num_pos):
            if i % 100 == 0:
                print("%010d    " % i, sep="", end="\r")
            grid_index = np.unravel_index(i, self.grid_pos.shape[0:3])
            probe_pos = self.grid_pos[tuple(grid_index)]
            target_density = self.density[tuple(grid_index)]
            atom_copy = self.atoms.copy()
            probe_atom = ase.atom.Atom(0, probe_pos)
            atom_copy.append(probe_atom)
            graphobj = FeatureGraphVirtual(atom_copy, "const", self.cutoff_radius, lambda x: x, density=target_density)
            yield graphobj
        print("")
