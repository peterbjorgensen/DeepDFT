import numpy as np
import os
import multiprocessing
import itertools
import ase
import warnings
import tarfile
import os
from ase.neighborlist import NeighborList
from ase.calculators.vasp import VaspChargeDensity
import msgnet

class FeatureGraphVirtual():
    def __init__(
        self,
        atoms_obj: ase.Atoms,
        cutoff_type,
        cutoff_radius,
        atom_to_node_fn,
        self_interaction=False,
        **kwargs
    ):
        self.atoms = atoms_obj

        if cutoff_type == "const":
            graph_tuple = self.atoms_to_graph_const_cutoff(
                self.atoms,
                cutoff_radius,
                atom_to_node_fn,
                self_interaction=self_interaction,
            )
            self.edge_labels = ["distance"]
        else:
            raise ValueError("cutoff_type not valid, given: %s" % cutoff_type)

        self.nodes, self.positions, self.conns, self.conns_offset, self.unitcell, self.probe_conns, self.probe_conns_offset = (
            graph_tuple
        )

        for key, val in kwargs.items():
            assert not hasattr(self, key), "Attribute %s is reserved" % key
            setattr(self, key, val)

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

        # Find the longest diagonal
        unitcell = atoms.get_cell()
        max_dist_squared = 0
        for a, b, c in itertools.product([-1,1], repeat=3):
            vec = a*unitcell[0] + b*unitcell[1] + c*unitcell[2]
            dist_squared = vec.dot(vec)
            max_dist_squared = max(max_dist_squared, dist_squared)

        max_dist = np.sqrt(max_dist_squared)

        if cutoff_covalent:
            radii = ase.data.covalent_radii[atom_numbers] * cutoff
        else:
            radii = []
            for at_num in atom_numbers:
                if at_num == 0:
                    radii.append(cutoff + max_dist)
                else:
                    radii.append(cutoff)
        neighborhood = NeighborList(
            radii, skin=0.0, self_interaction=self_interaction, bothways=True, primitive=ase.neighborlist.NewPrimitiveNeighborList
        )
        neighborhood.update(atoms)

        nodes = []
        connections = []
        connections_offset = []
        probe_connections = []
        probe_connections_offset = []
        if np.any(atoms.get_pbc()):
            atom_positions = atoms.get_positions(wrap=True)[:-1]
        else:
            atom_positions = atoms.get_positions(wrap=False)[:-1]

        for ii in range(len(atoms)-1):
            nodes.append(atom_to_node_fn(atom_numbers[ii]))

        for ii in range(len(atoms)):
            neighbor_indices, offset = neighborhood.get_neighbors(ii)
            for jj, offs in zip(neighbor_indices, offset):
                if atom_numbers[jj] == 0:
                    continue # The probe atom (number 0) has no outgoing connections
                elif atom_numbers[ii] == 0:
                    probe_connections.append([jj, 0]) # Probe atom is index zero because it is handled separately
                    probe_connections_offset.append([[offs[0], offs[1], offs[2]], [0,0,0]])
                else:
                    #ii_pos = atom_positions[ii]
                    #jj_pos = atom_positions[jj] + np.dot(offs, unitcell)
                    #dist_vec = ii_pos - jj_pos
                    #dist = np.sqrt(np.dot(dist_vec, dist_vec))
                    connections.append([jj, ii])
                    connections_offset.append([[offs[0], offs[1], offs[2]], [0,0,0]])

        return (
            np.array(nodes),
            atom_positions,
            np.array(connections),
            np.array(connections_offset),
            unitcell,
            np.array(probe_connections),
            np.array(probe_connections_offset),
        )

class VaspChargeDataLoader(msgnet.dataloader.DataLoader):
    def __init__(self, vasp_fname, cutoff_radius):
        super().__init__()
        self.basename = os.path.basename(vasp_fname)
        self.download_dest = vasp_fname
        self.cutoff_radius = cutoff_radius

    def _download_data(self):
        pass

    @property
    def final_dest(self):
        cutname = "%s-%.2f" % (self.cutoff_type, self.cutoff_radius)
        return "/%s/%s_%s_%s.pkz" % (
            msgnet.defaults.datadir,
            self.__class__.__name__,
            self.basename,
            cutname,
        )

    def _preprocess(self):
        with tarfile.open(self.download_dest, "r:*") as tar:
            for i, tarinfo in enumerate(tar.getmembers()):
                buf = tar.extractfile(tarinfo)
                tmppath = "/tmp/extracted%d" % os.getpid()
                with open(tmppath, "wb") as tmpfile:
                    tmpfile.write(buf.read())
                os.remove(tmppath)
                vasp_charge = VaspChargeDensity(filename=tmppath)
                density = vasp_charge.chg[-1] #seperate density
                atoms = vasp_charge.atoms[-1] #seperate atom positions
                probe_pos = np.array([0.5, 0.5, 0.5]).dot(atoms.get_cell())
                probe_atom = ase.atom.Atom(0, probe_pos)
                atoms.append(probe_atom)

                # Calculate grid positions
                ngridpts = np.array(density.shape) #grid matrix
                grid_pos = np.meshgrid(
                    np.arange(ngridpts[0])/density.shape[0],
                    np.arange(ngridpts[1])/density.shape[1],
                    np.arange(ngridpts[2])/density.shape[2],
                    indexing='ij',
                )
                grid_pos = np.stack(grid_pos, 3)
                grid_pos = np.dot(grid_pos, atoms.get_cell())

                graphobj = FeatureGraphVirtual(atoms, "const", self.cutoff_radius, lambda x: x, density=density, grid_position=grid_pos)
                yield graphobj
                if i % 100 == 0:
                    print("%010d    " % i, sep="", end="\r")
            print("")
