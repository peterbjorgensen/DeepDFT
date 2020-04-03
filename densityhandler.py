import multiprocessing
import sklearn.model_selection
import threading
import queue
import itertools
import math
import numpy as np

def set_len_to_segments(set_len):
    return np.repeat(np.arange(len(set_len)), set_len)

def get_folds(objects, folds):
    return [o for o in objects if o.fold in folds]

def train_queue_worker(index_generator, compressed_objects, queue, batch_size, probe_count):
    while True:
        training_dict = DensityDataHandler.sample_objects(index_generator, batch_size, probe_count, compressed_objects)
        queue.put(training_dict)

def thread_handover(mp_queue, thread_queue):
    while True:
        thread_queue.put(mp_queue.get())

class DataHandler:
    def __init__(self, graph_objects, graph_targets=["total_energy"]):
        self.graph_objects = graph_objects
        self.graph_targets = graph_targets
        self.train_index_generator = self.idx_epoch_gen(len(graph_objects))

    def get_train_batch(self, batch_size):
        rand_choice = itertools.islice(self.train_index_generator, batch_size)
        training_dict = self.list_to_matrices(
            [self.graph_objects[idx] for idx in rand_choice],
            graph_targets=self.graph_targets,
        )
        self.modify_dict(training_dict)
        return training_dict

    def get_test_batches(self, batch_size):
        num_test_batches = int(math.ceil(len(self.graph_objects) / batch_size))
        for batch_idx in range(num_test_batches):
            test_dict = self.list_to_matrices(
                self.graph_objects[
                    batch_idx * batch_size : (batch_idx + 1) * batch_size
                ],
                graph_targets=self.graph_targets,
            )
            self.modify_dict(test_dict)
            yield test_dict

    def __len__(self):
        return len(self.graph_objects)

    @staticmethod
    def idx_epoch_gen(num_objects):
        while 1:
            for n in np.random.permutation(num_objects):
                yield n

    @staticmethod
    def list_to_matrices(graph_list, graph_targets=["total_energy"]):
        """list_to_matrices
        Convert list of FeatureGraph objects to dictionary with concatenated properties

        :param graph_list:
        :return: dictionary of stacked vectors and matrices
        """
        nodes_created = 0
        all_nodes = []
        all_conn = []
        all_conn_offsets = []
        all_edges = []
        all_graph_targets = []
        all_X = []
        all_unitcells = []
        set_len = []
        edges_len = []
        for gr in graph_list:
            nodes, conn, conn_offset, edges, X, unitcell = (
                gr.nodes,
                gr.conns,
                gr.conns_offset,
                gr.edges,
                gr.positions,
                gr.unitcell,
            )
            conn_shifted = np.copy(conn) + nodes_created
            all_nodes.append(nodes)
            all_conn.append(conn_shifted)
            all_conn_offsets.append(conn_offset)
            all_unitcells.append(unitcell)
            all_edges.append(edges)
            all_graph_targets.append(np.array([getattr(gr, t) for t in graph_targets]))
            all_X.append(X)
            nodes_created += nodes.shape[0]
            set_len.append(nodes.shape[0])
            edges_len.append(edges.shape[0])
        cat = lambda x: np.concatenate(x, axis=0)
        outdict = {
            "nodes": cat(all_nodes),
            "nodes_xyz": cat(all_X),
            "edges": cat(all_edges),
            "connections": cat(all_conn),
            "connections_offsets": cat(all_conn_offsets),
            "graph_targets": np.vstack(all_graph_targets),
            "set_lengths": np.array(set_len),
            "unitcells": np.stack(all_unitcells, axis=0),
            "edges_lengths": np.array(edges_len),
        }
        outdict["segments"] = set_len_to_segments(outdict["set_lengths"])
        return outdict

    def get_normalization(self, per_atom=False):
        x_sum = np.zeros(len(self.graph_targets))
        x_2 = np.zeros(len(self.graph_targets))
        num_objects = 0
        for obj in self.graph_objects:
            for i, target in enumerate(self.graph_targets):
                x = getattr(obj, target)
                if per_atom:
                    x = x / obj.nodes.shape[0]
                x_sum[i] += x
                x_2[i] += x ** 2.0
                num_objects += 1
        # Var(X) = E[X^2] - E[X]^2
        x_mean = x_sum / num_objects
        x_var = x_2 / num_objects - (x_mean) ** 2.0

        return x_mean, np.sqrt(x_var)

    def train_test_split(
        self,
        split_type=None,
        num_folds=None,
        test_fold=None,
        validation_size=None,
        test_size=None,
        deterministic=True,
    ):
        if split_type == "count" or split_type == "fraction":
            if deterministic:
                random_state = 21
            else:
                random_state = None
            train, test = sklearn.model_selection.train_test_split(
                self.graph_objects, test_size=test_size, random_state=random_state
            )
        elif split_type == "fold":
            assert test_fold < num_folds
            assert test_fold >= 0
            train_folds = [i for i in range(num_folds) if i != test_fold]
            train, test = (
                get_folds(self.graph_objects, train_folds),
                get_folds(self.graph_objects, [test_fold]),
            )
        else:
            raise ValueError("Unknown split type %s" % split_type)

        if validation_size:
            if deterministic:
                random_state = 47
            else:
                random_state = None
            train, validation = sklearn.model_selection.train_test_split(
                train, test_size=validation_size, random_state=random_state
            )
        else:
            validation = []

        return self.from_self(train), self.from_self(test), self.from_self(validation)

    def modify_dict(self, train_dict):
        pass

    def from_self(self, objects):
        return self.__class__(objects, self.graph_targets)

class DensityDataHandler(DataHandler):
    def __init__(self, graph_objects, preprocessing_size=10, preprocessing_batch_size=1, preprocessing_probe_count=1000):
        self.preprocessing_size = preprocessing_size
        self.preprocessing_batch_size = preprocessing_batch_size
        self.preprocessing_probe_count = preprocessing_probe_count
        self.graph_objects = graph_objects
        self.train_queue = None
        self.train_index_generator = self.idx_epoch_gen(len(self.graph_objects))

    def setup_train_queue(self):
        self.mp_train_queue = multiprocessing.Queue(self.preprocessing_size)
        self.train_queue = queue.Queue(self.preprocessing_size)
        pargs = (
            self.train_index_generator,
            self.graph_objects,
            self.mp_train_queue,
            self.preprocessing_batch_size,
            self.preprocessing_probe_count,
            )
        self.train_worker = multiprocessing.Process(
            target=train_queue_worker,
            args=pargs,
            )
        self.thread_worker = threading.Thread(
            target=thread_handover,
            args=(self.mp_train_queue, self.train_queue),
        )
        self.thread_worker.start()
        self.train_worker.start()

    def from_self(self, objects):
        return self.__class__(
            objects,
            preprocessing_size=self.preprocessing_size,
            preprocessing_batch_size=self.preprocessing_batch_size,
            preprocessing_probe_count=self.preprocessing_probe_count)

    def get_train_batch(self, batch_size, probe_count=1000):
        if self.train_queue is not None:
            assert self.preprocessing_batch_size == batch_size, "Train batch size is fixed when using train queue"
            assert self.preprocessing_probe_count == probe_count, "Probe count is fixed when using train queue"
            training_dict = self.train_queue.get()
            if self.train_queue.qsize() < (self.preprocessing_size // 2):
                try:
                    self.train_queue.put(training_dict, block=False)
                except queue.Full:
                    pass
        else:
            training_dict = self.sample_objects(self.train_index_generator, batch_size, probe_count, self.graph_objects)
        self.modify_dict(training_dict)
        return training_dict

    @staticmethod
    def sample_objects(index_generator, batch_size, probe_count, graph_objects):
        rand_choice = itertools.islice(index_generator, batch_size)
        graph_objects = [graph_objects[idx] for idx in rand_choice]
        graph_objects = list(map(lambda x: x.decompress() if hasattr(x, "decompress") else x, graph_objects))

        probe_choice_max = [np.prod(gobj.grid_position.shape[0:3]) for gobj in graph_objects]
        probe_choice = [np.random.randint(probe_max, size=probe_count) for probe_max in probe_choice_max]
        probe_choice = [np.unravel_index(pchoice, gobj.grid_position.shape[0:3]) for pchoice, gobj in zip(probe_choice, graph_objects)]
        probe_pos = [gobj.grid_position[pchoice] for pchoice, gobj in zip(probe_choice, graph_objects)]
        probe_target = [gobj.density[pchoice] for pchoice, gobj in zip(probe_choice, graph_objects)]
        training_dict = DensityDataHandler.list_to_matrices(graph_objects, probe_pos, probe_target)
        return training_dict

    def get_test_batches(self, probe_count=100, decimation=1):
        global_slice_index = 0
        for cobj in self.graph_objects:
            if hasattr(cobj, "decompress"):
                gobj = cobj.decompress()
            else:
                gobj = cobj
            num_positions = np.prod(gobj.grid_position.shape[0:3])
            num_slices = int(math.ceil(num_positions / probe_count))
            for slice_index in range(num_slices):
                global_slice_index += 1
                if (global_slice_index % decimation) != 0:
                    continue
                flat_index = np.arange(slice_index*probe_count, min((slice_index+1)*probe_count, num_positions))
                pos_index = np.unravel_index(flat_index, gobj.grid_position.shape[0:3])
                target_density = gobj.density[pos_index]
                probe_pos = gobj.grid_position[pos_index]
                test_dict = self.list_to_matrices([gobj], [probe_pos], [target_density])

                self.modify_dict(test_dict)
                yield test_dict

    @staticmethod
    def list_to_matrices(graph_list, probe_pos_list, probe_target_list=None):
        """list_to_matrices
        Convert list of FeatureGraph objects to dictionary with concatenated properties

        :param graph_list:
        :return: dictionary of stacked vectors and matrices
        """
        nodes_created = 0
        probe_nodes_created = 0
        all_nodes = []
        all_conn = []
        all_conn_offsets = []
        all_probe_conn = []
        all_probe_conn_offsets = []
        all_xyz = []
        all_unitcells = []
        set_len = []
        edges_len = []
        for gr in graph_list:
            conn_shifted = np.copy(gr.conns) + nodes_created
            probe_conn_shifted = np.copy(gr.probe_conns) + np.array([[nodes_created, probe_nodes_created]], dtype=int)
            all_nodes.append(gr.nodes)
            all_conn.append(conn_shifted)
            all_conn_offsets.append(gr.conns_offset)
            all_probe_conn.append(probe_conn_shifted)
            all_probe_conn_offsets.append(gr.probe_conns_offset)
            all_unitcells.append(gr.unitcell)
            all_xyz.append(gr.positions)
            nodes_created += gr.nodes.shape[0]
            probe_nodes_created += 1
            set_len.append(gr.nodes.shape[0])
            edges_len.append(gr.conns.shape[0])
        cat = lambda x: np.concatenate(x, axis=0)
        outdict = {
            "nodes": cat(all_nodes),
            "nodes_xyz": cat(all_xyz),
            "connections": cat(all_conn),
            "connections_offset": cat(all_conn_offsets),
            "probes_xyz": np.stack(probe_pos_list, axis=0),
            "probes_target": np.stack(probe_target_list, axis=0),
            "probes_connection": cat(all_probe_conn),
            "probes_connection_offset": cat(all_probe_conn_offsets),
            "set_lengths": np.array(set_len),
            "unitcells": np.stack(all_unitcells, axis=0),
            "edges_lengths": np.array(edges_len),
        }
        outdict["segments"] = set_len_to_segments(outdict["set_lengths"])
        outdict["edges_segments"] = set_len_to_segments(outdict["edges_lengths"])
        return outdict
