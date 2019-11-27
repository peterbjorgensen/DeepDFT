import os
import glob
import numpy as np
import ase
import tensorflow as tf
from tensorflow.contrib import layers
import msgnet
import interpolation

def repeat(data, n_repeats, axis):
    assert axis==0
    multiples = tf.stack((1, n_repeats), 0)
    final_shape = tf.stack((n_repeats, 1)) * tf.shape(data)
    tiled = tf.tile(data, multiples=multiples)
    repeated = tf.reshape(tiled, final_shape)
    return repeated

def compute_messages(
    nodes,
    conn,
    edges,
    message_fn,
    act_fn,
    distances=None,
    receiver_nodes=None,
    include_receiver=True,
    include_sender=True,
    only_messages=False,
    mean_messages=True,
):
    """
    :param nodes: (n_nodes, n_node_features) tensor of nodes, float32.
    :param conn: (n_edges, 2) tensor of indices indicating an edge between nodes at those indices, [from, to] int32.
    :param edges: (n_edges, n_edge_features) tensor of edge features, float32.
    :param message_fn: message function, will be called with two inputs with shapes (n_edges, K*n_node_features), (n_edges,n_edge_features), where K is 2 if include_receiver=True and 1 otherwise, and must return a tensor of size (n_edges, n_output)
    :param act_fn: A pointwise activation function applied after the sum.
    :param reveiver_nodes: (n_receive_nodes, n_node_features) If given, use these nodes for receiver
    :param include_receiver: Include receiver node in computation of messages.
    :param include_sender: Include sender node in computation of messages.
    :param only_messages: Do not sum up messages
    :return: (n_edges, n_output) if only_messages is True, otherwise (n_nodes, n_output) Sum of messages arriving at each node.
    """
    if receiver_nodes is not None:
        n_nodes = tf.shape(receiver_nodes)[0]
    else:
        n_nodes = tf.shape(nodes)[0]

    n_node_features = nodes.get_shape()[1].value
    n_edge_features = edges.get_shape()[1].value

    if include_receiver and include_sender:
        # Use both receiver and sender node features in message computation
        if receiver_nodes is not None:
            from_nodes = tf.gather(nodes, conn[:, 0])
            to_nodes = tf.gather(receiver_nodes, conn[:, 1])
            message_inputs = tf.concat((from_nodes, to_nodes), axis=1)
        else:
            message_inputs = tf.gather(nodes, conn)  # n_edges, 2, n_node_features
            message_inputs = tf.reshape(message_inputs, (-1, 2 * n_node_features))
    elif include_sender:  # Only use sender node features (index=0)
        message_inputs = tf.gather(nodes, conn[:, 0])  # n_edges, n_node_features
    elif include_receiver:  # Only use receiver node features (index=1)
        if receiver_nodes is not None:
            message_inputs = tf.gather(receiver_nodes, conn[:, 1])
        else:
            message_inputs = tf.gather(nodes, conn[:, 1])  # n_edges, n_node_features
    else:
        raise ValueError(
            "Messages must include at least one of sender and receiver nodes"
        )
    if distances is not None:
        messages = message_fn(message_inputs, edges, distances)  # n_edges, n_output
    else:
        messages = message_fn(message_inputs, edges)  # n_edges, n_output

    if only_messages:
        return messages

    idx_dest = conn[:, 1]
    if mean_messages:
        # tf.bincount not supported on GPU in TF 1.4, so do this instead
        count = tf.unsorted_segment_sum(
            tf.ones_like(idx_dest, dtype=tf.float32), idx_dest, n_nodes
        )
        count = tf.maximum(count, 1)  # Avoid division by zero
        msg_pool = tf.unsorted_segment_sum(
            messages, idx_dest, n_nodes
        ) / tf.expand_dims(count, -1)
    else:
        msg_pool = tf.unsorted_segment_sum(messages, idx_dest, n_nodes)
    return act_fn(msg_pool)


def create_msg_function(num_outputs, cutoff_func=None, **kwargs):
    def func(nodes, edges, distances=None):
        tf.add_to_collection("msg_input_nodes", nodes)
        tf.add_to_collection("msg_input_edges", edges)
        with tf.variable_scope("gates"):
            gates = msgnet.defaults.mlp(
                edges,
                [num_outputs, num_outputs],
                last_activation=msgnet.defaults.nonlinearity,
                activation=msgnet.defaults.nonlinearity,
                weights_initializer=msgnet.defaults.initializer,
            )
            tf.add_to_collection("msg_gates", gates)
        with tf.variable_scope("pre"):
            pre = layers.fully_connected(
                nodes,
                num_outputs,
                activation_fn=tf.identity,
                weights_initializer=msgnet.defaults.initializer,
                biases_initializer=None,
                **kwargs
            )
            tf.add_to_collection("msg_pregates", pre)
        if distances is not None:
            assert cutoff_func is not None
            gates = gates * cutoff_func(distances)
        output = pre * gates
        tf.add_to_collection("msg_outputs", output)
        return output

    return func


def edge_update(node_states, edge_states):
    edge_states_len = int(edge_states.get_shape()[1])
    nodes_states_len = int(node_states.get_shape()[1])
    combined = tf.concat((node_states, edge_states), axis=1)
    new_edge = msgnet.defaults.mlp(
        combined,
        [nodes_states_len, nodes_states_len // 2],
        activation=msgnet.defaults.nonlinearity,
        weights_initializer=msgnet.defaults.initializer,
    )
    return new_edge

class BaselineModel():

    def __init__(self, single_atom_dir):
        self.x_min, self.x_max, self.ref_matrix = self.build_model(single_atom_dir)

    def build_model(self, atom_dir):
        interp_funcs = {}
        for fname in glob.glob(os.path.join(atom_dir, "*.txt")):
            head, tail = os.path.split(fname)
            sym, ext = os.path.splitext(tail)

            # The density files are in Bohr and electrons/Bohr**3
            # Convert to Å and electrons/Å**3
            data = np.loadtxt(fname, dtype=np.float32)
            data[:, 0] = data[:, 0]*ase.units.Bohr
            data[:, 1] = data[:, 1]/ase.units.Bohr**3 # SCF density
            data[:, 2] = data[:, 2]/ase.units.Bohr**3 # Core density
            #integral = scipy.integrate.trapz(data[:,1]*data[:,0]**2, data[:,0])*4*np.pi
            #print(fname, integral)

            x_step = np.diff(data[:,0])
            x_min = data[0,0]
            x_max = data[-1,0]
            x_step_mean = np.mean(x_step)
            assert np.max(x_step-x_step_mean) < 1e-6, "regular grid assumed"
            interp_funcs[ase.data.atomic_numbers[sym]] = data[:,1]-data[:,2] # Subtract core density
        for i, (key, val) in enumerate(interp_funcs.items()):
            if i == 0:
                ref_len = val.shape[0]
                ref_matrix = np.zeros((len(ase.data.chemical_symbols), ref_len), dtype=np.float32)
            ref_matrix[key] = val
        return x_min, x_max, tf.convert_to_tensor(ref_matrix)


    def get_density(self, atomic_numbers, distances):
        """get_density

        :param atomic_numbers: 1-D tensor of length num_nodes
        :param distances: [N,1] tensor of length num_probes_edges with distance between each node and the corresponding probes
        """
        import interpolation
        y_ref_mat = tf.gather(self.ref_matrix, atomic_numbers) # num_nodes, num_ref_values
        density = interpolation.batch_interp_regular_1d_grid(distances, self.x_min, self.x_max, y_ref_mat)

        return density

class DensityMsgPassing:
    def __init__(
        self,
        n_node_features=1,
        n_edge_features=1,
        num_passes=3,
        embedding_shape=None,
        edge_feature_expand=None,
        msg_share_weights=False,
        use_edge_updates=False,
        edge_output_fn=None,
        avg_msg=False,
        target_mean=0.0,
        target_std=1.0,
        hard_cutoff=6.0,
        single_atom_reference_dir=None
    ):
        """__init__

        :param n_node_features:
        :param n_edge_features:
        :param n_graph_target_features:
        :param num_passes:
        :param embedding_shape: (num_species, embedding_size)
        :param edge_feature_expand: (start, step, end)
        """

        # Symbolic input variables
        if embedding_shape is not None:
            self.sym_nodes = tf.placeholder(np.int32, shape=(None,), name="sym_nodes")
        else:
            self.sym_nodes = tf.placeholder(
                np.float32, shape=(None, n_node_features), name="sym_nodes"
            )
        self.sym_nodes_xyz = tf.placeholder(np.float32, shape=(None, 3), name="sym_nodes_xyz")

        self.sym_edges = tf.placeholder(
            np.float32, shape=(None, n_edge_features), name="sym_edges"
        )
        self.edge_output_fn = edge_output_fn
        self.sym_conn = tf.placeholder(np.int32, shape=(None, 2), name="sym_conn")
        self.sym_conn_offset = tf.placeholder(np.float32, shape=(None, 2, 3), name="sym_conn_offset")
        self.sym_segments = tf.placeholder(
            np.int32, shape=(None,), name="sym_segments_map"
        )
        self.sym_edges_segments = tf.placeholder(
            np.int32, shape=(None,), name="sym_edges_segments"
        )
        self.sym_set_len = tf.placeholder(np.int32, shape=(None,), name="sym_set_len")
        self.sym_edges_len = tf.placeholder(np.int32, shape=(None,), name="sym_edges_len")
        self.sym_probe_conn = tf.placeholder(np.int32, shape=(None, 2), name="sym_probe_conn")
        self.sym_probe_conn_offset = tf.placeholder(np.float32, shape=(None, 2, 3), name="sym_probe_conn_offset")
        self.sym_probe_xyz = tf.placeholder(np.float32, shape=(None, None, 3), name="sym_probe_xyz")
        self.sym_unitcells = tf.placeholder(np.float32, shape=(None, 3, 3), name="unitcells")

        self.input_symbols = {
            "nodes": self.sym_nodes,
            "nodes_xyz": self.sym_nodes_xyz,
            "connections": self.sym_conn,
            "connections_offset": self.sym_conn_offset,
            "probes_xyz": self.sym_probe_xyz,
            "probes_connection": self.sym_probe_conn,
            "probes_connection_offset": self.sym_probe_conn_offset,
            "segments": self.sym_segments,
            "unitcells": self.sym_unitcells,
            "set_lengths": self.sym_set_len,
            "edges_segments": self.sym_edges_segments,
        }

        unitcells = tf.gather(self.sym_unitcells, self.sym_edges_segments)
        node_offset = tf.matmul(self.sym_conn_offset, unitcells)
        node_unitcell_pos = tf.gather(self.sym_nodes_xyz, self.sym_conn)
        conn_pos = node_unitcell_pos + node_offset
        conn_diff = conn_pos[:, 1] - conn_pos[:, 0]
        conn_dist = tf.sqrt(tf.reduce_sum(tf.square(conn_diff), axis=-1)) # edge_count

        probe_unitcells = tf.gather(self.sym_unitcells, self.sym_probe_conn[:, 1])
        node_offset = tf.matmul(tf.expand_dims(self.sym_probe_conn_offset[:, 0], 1), probe_unitcells)
        node_unitcell_pos = tf.gather(self.sym_nodes_xyz, self.sym_probe_conn[:, 0])
        conn_pos = node_unitcell_pos + tf.squeeze(node_offset, axis=1)
        probe_unitcell_pos = tf.gather(self.sym_probe_xyz, self.sym_probe_conn[:, 1]) # edge_count, probe_count, 3
        conn_pos = tf.expand_dims(conn_pos, 1) # edge_count, 1, 3
        conn_diff = conn_pos - probe_unitcell_pos
        probe_dist = tf.sqrt(tf.reduce_sum(tf.square(conn_diff), axis=-1)) # probe_edge_count, probe_count
        probe_count = tf.shape(probe_dist)[1]
        probe_dist_flat = tf.reshape(probe_dist, (-1,1)) # probe_edge_count x probe_count
        repeated_conns = repeat(self.sym_probe_conn, probe_count, 0)
        repeated_conns_from = repeated_conns[:, 0] # probe_edge_count x probe_count
        repeated_conns_to = repeated_conns[:, 1] * probe_count
        repeated_conns_to = repeated_conns_to + tf.tile(tf.range(0, probe_count, dtype=tf.int32), [tf.shape(self.sym_probe_conn)[0]]) # probe_edge_count x probe_count
        repeated_conns = tf.stack((repeated_conns_from, repeated_conns_to), 1)

        probe_conn_mask = tf.less(tf.squeeze(probe_dist_flat, axis=1), hard_cutoff)
        sym_probe_dist = tf.boolean_mask(probe_dist_flat, probe_conn_mask, axis=0)
        sym_probe_conn = tf.boolean_mask(repeated_conns, probe_conn_mask, axis=0)


        # Setup constants for normalizing/denormalizing graph level outputs
        self.sym_target_mean = tf.get_variable(
            "target_mean",
            dtype=tf.float32,
            shape=[],
            trainable=False,
            initializer=tf.constant_initializer(target_mean),
        )
        self.sym_target_std = tf.get_variable(
            "target_std",
            dtype=tf.float32,
            shape=[],
            trainable=False,
            initializer=tf.constant_initializer(target_std),
        )

        sym_edges = tf.expand_dims(conn_dist, 1)
        if edge_feature_expand is not None:
            init_edges = msgnet.utilities.gaussian_expansion(
                sym_edges, edge_feature_expand
            )
        else:
            init_edges = sym_edges

        if embedding_shape is not None:
            # Setup embedding matrix
            stddev = np.sqrt(1.0 / np.sqrt(embedding_shape[1]))
            self.species_embedding = tf.Variable(
                initial_value=np.random.standard_normal(embedding_shape) * stddev,
                trainable=True,
                dtype=np.float32,
                name="species_embedding_matrix",
            )
            hidden_state0 = tf.gather(self.species_embedding, self.sym_nodes)
        else:
            hidden_state0 = self.sym_nodes

        hidden_state = hidden_state0

        hidden_state_len = int(hidden_state.get_shape()[1])

        # Setup edge update function
        if use_edge_updates:
            edge_msg_fn = edge_update
            edges = compute_messages(
                hidden_state,
                self.sym_conn,
                init_edges,
                edge_msg_fn,
                tf.identity,
                include_receiver=True,
                include_sender=True,
                only_messages=True,
            )
        else:
            edges = init_edges

        # Setup interaction messages
        soft_cutoff_func = lambda x: 1.-tf.sigmoid(5*(x-(hard_cutoff-1.5)))
        msg_fn = create_msg_function(hidden_state_len, cutoff_func=soft_cutoff_func)
        act_fn = tf.identity
        hidden_states_list = []
        for i in range(num_passes):
            if msg_share_weights:
                scope_suffix = ""
                reuse = i > 0
            else:
                scope_suffix = "%d" % i
                reuse = False
            with tf.variable_scope("msg" + scope_suffix, reuse=reuse):
                sum_msg = compute_messages(
                    hidden_state,
                    self.sym_conn,
                    edges,
                    msg_fn,
                    act_fn,
                    distances=sym_edges,
                    include_receiver=True,
                    mean_messages=avg_msg,
                )
            with tf.variable_scope("update" + scope_suffix, reuse=reuse):
                hidden_state += msgnet.defaults.mlp(
                    sum_msg,
                    [hidden_state_len, hidden_state_len],
                    activation=msgnet.defaults.nonlinearity,
                    weights_initializer=msgnet.defaults.initializer,
                )
                hidden_states_list.append(hidden_state)
            with tf.variable_scope("edge_update" + scope_suffix, reuse=reuse):
                if use_edge_updates and (i < (num_passes - 1)):
                    edges = compute_messages(
                        hidden_state,
                        self.sym_conn,
                        edges,
                        edge_msg_fn,
                        tf.identity,
                        include_receiver=True,
                        include_sender=True,
                        only_messages=True,
                    )

        # Setup probe messages
        zeros_dims = tf.stack([tf.shape(self.sym_probe_xyz)[0]*tf.shape(self.sym_probe_xyz)[1], embedding_shape[1]])
        probe_state = tf.fill(zeros_dims, 0.0)

        probe_edges = sym_probe_dist
        if edge_feature_expand is not None:
            probe_edges = msgnet.utilities.gaussian_expansion(
                probe_edges, edge_feature_expand
            )
        for i in range(num_passes):
            if msg_share_weights:
                scope_suffix = ""
                reuse = i > 0
            else:
                scope_suffix = "%d" % i
                reuse = False
            with tf.variable_scope("probe_msg" + scope_suffix, reuse=reuse):
                sum_msg = compute_messages(
                    hidden_states_list[i],
                    sym_probe_conn,
                    probe_edges,
                    msg_fn,
                    act_fn,
                    distances=sym_probe_dist,
                    receiver_nodes=probe_state,
                    include_receiver=True,
                    mean_messages=avg_msg,
                )
            with tf.variable_scope("probe_update" + scope_suffix, reuse=reuse):
                gates = msgnet.defaults.mlp(
                    probe_state,
                    [hidden_state_len, hidden_state_len],
                    activation=msgnet.defaults.nonlinearity,
                    last_activation=tf.sigmoid,
                    weights_initializer=msgnet.defaults.initializer,
                )
                probe_state = probe_state*gates + (1.-gates)*msgnet.defaults.mlp(
                    sum_msg,
                    [hidden_state_len, hidden_state_len],
                    activation=msgnet.defaults.nonlinearity,
                    weights_initializer=msgnet.defaults.initializer,
                )
            #with tf.variable_scope("probe_edge_update" + scope_suffix, reuse=reuse):
                #if use_edge_updates and (i < (num_passes - 1)):
                    #probe_edges = compute_messages(
                        #hidden_states_list[i],
                        #sym_probe_conn,
                        #probe_edges,
                        #edge_msg_fn,
                        #tf.identity,
                        #receiver_nodes=probe_state,
                        #include_receiver=True,
                        #include_sender=True,
                        #only_messages=True,
                    #)

        self.nodes_out = probe_state
        # Readout probe_state
        density = msgnet.defaults.mlp(
            probe_state,
            [hidden_state_len, hidden_state_len, hidden_state_len, 1],
            activation=msgnet.defaults.nonlinearity,
            weights_initializer=msgnet.defaults.initializer,
        )

        if single_atom_reference_dir is not None:
            # Compute contribution from baseline model
            baseline = BaselineModel(single_atom_reference_dir)
            atomic_numbers = tf.gather(self.sym_nodes, sym_probe_conn[:,0])
            baseline_atom_density = baseline.get_density(atomic_numbers, sym_probe_dist)
            baseline_total_density = tf.unsorted_segment_sum(
                baseline_atom_density,
                sym_probe_conn[:,1],
                tf.shape(density)[0],
            )

            baseline_density_normalized = (baseline_total_density-self.sym_target_mean)/self.sym_target_std

            density = density + baseline_density_normalized

        self.graph_out_normalized = tf.reshape(density, tf.shape(self.sym_probe_xyz)[0:2], name="graph_out_normalized")
        self.graph_out = tf.add(self.graph_out_normalized * self.sym_target_std, self.sym_target_mean, name="graph_out")

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=24, max_to_keep=3)

    def save(self, session, destination, global_step):
        self.saver.save(session, destination, global_step=global_step)

    def load(self, session, path):
        self.saver.restore(session, path)

    def get_nodes_out(self):
        return self.nodes_out

    def get_graph_out(self):
        return self.graph_out

    def get_graph_out_normalized(self):
        return self.graph_out_normalized

    def get_normalization(self):
        return self.sym_target_mean, self.sym_target_std

    def get_input_symbols(self):
        return self.input_symbols
