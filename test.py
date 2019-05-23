import os
import timeit
import logging
import warnings
import numpy as np
import ase
import ase.io
import msgnet
import multiprocessing
import queue
import tensorflow as tf
from ase.calculators.vasp import VaspChargeDensity
from ase.neighborlist import NeighborList
from msgnet.dataloader import FeatureGraph

CUTOFF_ANGSTROM = 5.0

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

def data_worker(density, atoms, grid_pos, batch_size, queue):
    QUEUE_TTL = 10000
    while True:
        graph_obj_list = []
        num_pos = np.prod(grid_pos.shape[0:3])
        selection = np.random.randint(0, num_pos, batch_size)
        selection = np.unravel_index(selection, grid_pos.shape[0:3])
        for grid_index in np.stack(selection,-1):
            probe_pos = grid_pos[tuple(grid_index)]
            target_density = density[tuple(grid_index)]
            atom_copy = atoms.copy()
            probe_atom = ase.atom.Atom(0, probe_pos)
            atom_copy.append(probe_atom)
            graphobj = FeatureGraphVirtual(atom_copy, "const", CUTOFF_ANGSTROM, lambda x: x, density=target_density)
            graph_obj_list.append(graphobj)
        handler = msgnet.datahandler.EdgeSelectDataHandler(graph_obj_list, ["density"], [0])
        batch = next(handler.get_test_batches(batch_size))
        queue.put((batch,QUEUE_TTL))

class DataWrapper:
    QUEUE_SIZE = 10
    def __init__(self, vasp_charge, batch_size):
        self.vasp_charge = vasp_charge
        self.batch_size = batch_size
        self.density = vasp_charge.chg[-1] #seperate density
        self.atoms = vasp_charge.atoms[-1] #seperate atom positions
        ngridpts = np.array(self.density.shape) #grid matrix

        grid_pos = np.meshgrid(
            np.arange(ngridpts[0])/self.density.shape[0],
            np.arange(ngridpts[1])/self.density.shape[1],
            np.arange(ngridpts[2])/self.density.shape[2],
            indexing='ij',
        )
        grid_pos = np.stack(grid_pos, 3)
        self.grid_pos = np.dot(grid_pos, self.atoms.get_cell())

        self.train_queue = multiprocessing.Queue(self.QUEUE_SIZE)
        self.data_worker = multiprocessing.Process(target=data_worker, args=(self.density, self.atoms, self.grid_pos, self.batch_size, self.train_queue))
        self.data_worker.daemon = True
        self.data_worker.start()
        print(self.data_worker, self.data_worker.is_alive())

    def get_train_batch(self, batch_size):
        assert batch_size==self.batch_size, "Wrapper does not support flexible batch sizes"
        batch, ttl = self.train_queue.get()
        if ttl > 0:
            try:
                self.train_queue.put_nowait((batch, ttl-1))
            except queue.Full:
                pass
        return batch
        #graph_obj_list = []
        #num_pos = np.prod(self.grid_pos.shape[0:3])
        #selection = np.random.randint(0, num_pos, batch_size)
        #selection = np.unravel_index(selection, self.grid_pos.shape[0:3])
        #for grid_index in np.stack(selection,-1):
        #    probe_pos = self.grid_pos[tuple(grid_index)]
        #    target_density = self.density[tuple(grid_index)]
        #    atom_copy = self.atoms.copy()
        #    probe_atom = ase.atom.Atom(0, probe_pos)
        #    atom_copy.append(probe_atom)
        #    graphobj = FeatureGraphVirtual(atom_copy, "const", CUTOFF_ANGSTROM, lambda x: x, density=target_density)
        #    graph_obj_list.append(graphobj)
        #handler = msgnet.datahandler.EdgeSelectDataHandler(graph_obj_list, ["density"], [0])
        #return next(handler.get_test_batches(batch_size))

    def get_test_batches(self, batch_size):
        return [self.get_train_batch(batch_size)]

    def __len__(self):
        #return np.prod(self.density.shape)
        return 200

class ReadoutLastnode(msgnet.readout.ReadoutFunction):
    is_sum = False

    def __call__(self, nodes, segments):
        nodes_size = int(nodes.get_shape()[1])
        set_len = tf.segment_sum(tf.ones_like(segments), segments, name="set_len") - 1
        last_node_idx = tf.cumsum(set_len)
        last_nodes = tf.gather(nodes, last_node_idx)
        graph_out = msgnet.defaults.mlp(
            last_nodes,
            [nodes_size, nodes_size, self.output_size],
            activation=msgnet.defaults.nonlinearity,
            weights_initializer=msgnet.defaults.initializer,
        )
        return graph_out

def main():
    vasp_charge = VaspChargeDensity(filename="si30/CHGCAR")


    batch_size = 20
    embedding_size = 10

    datawrapper = DataWrapper(vasp_charge, batch_size)
    breakpoint()

    model = msgnet.MsgpassingNetwork(
        embedding_shape=(len(ase.data.chemical_symbols), embedding_size),
        edge_feature_expand=[(0, 0.1, CUTOFF_ANGSTROM+1)],
        use_edge_updates=False,
        readout_fn=ReadoutLastnode())

    trainer = msgnet.train.GraphOutputTrainer(model, datawrapper, batch_size=batch_size)

    num_steps = int(1e6)
    start_step = 0
    log_interval = 200
    val_obj = None
    train_obj = datawrapper

    start_time = timeit.default_timer()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Print shape of all trainable variables
        trainable_vars = tf.trainable_variables()
        for var, val in zip(trainable_vars, sess.run(trainable_vars)):
            logging.debug("%s %s", var.name, var.get_shape())

        for update_step in range(start_step, num_steps):
            trainer.step(sess, update_step)

            if (update_step % log_interval == 0) or (update_step + 1) == num_steps:
                test_start_time = timeit.default_timer()

                # Evaluate training set
                train_metrics = trainer.evaluate_metrics(
                    sess, train_obj, prefix="train"
                )

                # Evaluate validation set
                if val_obj:
                    val_metrics = trainer.evaluate_metrics(sess, val_obj, prefix="val")
                else:
                    val_metrics = {}

                all_metrics = {**train_metrics, **val_metrics}
                metric_string = " ".join(
                    ["%s=%f" % (key, val) for key, val in all_metrics.items()]
                )

                end_time = timeit.default_timer()
                test_end_time = timeit.default_timer()
                logging.info(
                    "t=%.1f (%.1f) %d %s lr=%f",
                    end_time - start_time,
                    test_end_time - test_start_time,
                    update_step,
                    metric_string,
                    trainer.get_learning_rate(update_step),
                )
                start_time = timeit.default_timer()

                # Do early stopping using validation data (if available)
                if val_obj:
                    if all_metrics["val_mae"] < best_val_mae:
                        model.save(
                            sess, logs_path + "model.ckpt", global_step=update_step
                        )
                        best_val_mae = all_metrics["val_mae"]
                        best_val_step = update_step
                        logging.info(
                            "best_val_mae=%f, best_val_step=%d",
                            best_val_mae,
                            best_val_step,
                        )
                    if (update_step - best_val_step) > 1e6:
                        logging.info(
                            "best_val_mae=%f, best_val_step=%d",
                            best_val_mae,
                            best_val_step,
                        )
                        logging.info("No improvement in last 1e6 steps, stopping...")
                        model.save(
                            sess, logs_path + "model.ckpt", global_step=update_step
                        )
                        return
                else:
                    model.save(sess, logs_path + "model.ckpt", global_step=update_step)


if __name__ == "__main__":
    logs_path = "logs/"
    os.makedirs(logs_path, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(logs_path + "printlog.txt", mode="w"),
            logging.StreamHandler(),
        ],
    )
    main()
