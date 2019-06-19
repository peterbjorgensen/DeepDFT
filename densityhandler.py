import itertools
import math
import numpy as np
import msgnet

class DensityDataHandler(msgnet.datahandler.DataHandler):
    def __init__(self, graph_objects):
        self.graph_objects = graph_objects
        self.train_index_generator = self.idx_epoch_gen(len(graph_objects))

    def from_self(self, objects):
        return self.__class__(objects)

    def get_train_batch(self, batch_size, probe_count=100):
        graph_choice = itertools.islice(self.train_index_generator, batch_size)
        graph_objects = [self.graph_objects[idx] for idx in graph_choice]
        probe_choice_max = [np.prod(gobj.grid_position.shape[0:3]) for gobj in graph_objects]
        probe_choice = [np.random.randint(probe_max, size=probe_count) for probe_max in probe_choice_max]
        probe_choice = [np.unravel_index(pchoice, gobj.grid_position.shape[0:3]) for pchoice, gobj in zip(probe_choice, graph_objects)]
        probe_pos = [gobj.grid_position[pchoice] for pchoice, gobj in zip(probe_choice, graph_objects)]
        probe_target = [gobj.density[pchoice] for pchoice, gobj in zip(probe_choice, graph_objects)]
        training_dict = self.list_to_matrices(graph_objects, probe_pos, probe_target)
        self.modify_dict(training_dict)
        return training_dict

    def get_test_batches(self, probe_count=100):
        for gobj in self.graph_objects:
            num_positions = np.prod(gobj.grid_position.shape[0:3])
            num_slices = int(math.ceil(num_positions / probe_count))
            for slice_index in range(num_slices):
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
        outdict["segments"] = msgnet.datahandler.set_len_to_segments(outdict["set_lengths"])
        outdict["edges_segments"] = msgnet.datahandler.set_len_to_segments(outdict["edges_lengths"])
        return outdict
