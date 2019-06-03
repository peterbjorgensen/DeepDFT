import msgnet
import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers

def compute_messages(
    nodes,
    conn,
    edges,
    message_fn,
    act_fn,
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
    :param include_receiver: Include receiver node in computation of messages.
    :param include_sender: Include sender node in computation of messages.
    :param only_messages: Do not sum up messages
    :return: (n_edges, n_output) if only_messages is True, otherwise (n_nodes, n_output) Sum of messages arriving at each node.
    """
    n_nodes = tf.shape(nodes)[0]
    n_node_features = nodes.get_shape()[1].value
    n_edge_features = edges.get_shape()[1].value

    if include_receiver and include_sender:
        # Use both receiver and sender node features in message computation
        message_inputs = tf.gather(nodes, conn)  # n_edges, 2, n_node_features
        reshaped = tf.reshape(message_inputs, (-1, 2 * n_node_features))
    elif include_sender:  # Only use sender node features (index=0)
        message_inputs = tf.gather(nodes, conn[:, 0])  # n_edges, n_node_features
        reshaped = message_inputs
    elif include_receiver:  # Only use receiver node features (index=1)
        message_inputs = tf.gather(nodes, conn[:, 1])  # n_edges, n_node_features
        reshaped = message_inputs
    else:
        raise ValueError(
            "Messages must include at least one of sender and receiver nodes"
        )
    messages = message_fn(reshaped, edges)  # n_edges, n_output

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


def create_msg_function(num_outputs, **kwargs):
    def func(nodes, edges):
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
        readout_fn=None,
        edge_output_fn=None,
        avg_msg=False,
        target_mean=0.0,
        target_std=1.0,
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
        self.sym_edges = tf.placeholder(
            np.float32, shape=(None, n_edge_features), name="sym_edges"
        )
        self.readout_fn = readout_fn
        self.edge_output_fn = edge_output_fn
        self.sym_conn = tf.placeholder(np.int32, shape=(None, 2), name="sym_conn")
        self.sym_segments = tf.placeholder(
            np.int32, shape=(None,), name="sym_segments_map"
        )
        self.sym_set_len = tf.placeholder(np.int32, shape=(None,), name="sym_set_len")

        self.input_symbols = {
            "nodes": self.sym_nodes,
            "edges": self.sym_edges,
            "connections": self.sym_conn,
            "segments": self.sym_segments,
            "set_lengths": self.sym_set_len,
        }

        sym_conn_dest = tf.gather(self.sym_nodes, self.sym_conn[:,1])
        sym_conn_is_special = tf.equal(sym_conn_dest, 0)

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

        if edge_feature_expand is not None:
            init_edges = msgnet.utilities.gaussian_expansion(
                self.sym_edges, edge_feature_expand
            )
        else:
            init_edges = self.sym_edges

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

        sym_conn_normal = tf.boolean_mask(self.sym_conn, tf.logical_not(sym_conn_is_special))
        sym_conn_special = tf.boolean_mask(self.sym_conn, sym_conn_is_special)
        init_edges_normal = tf.boolean_mask(init_edges, tf.logical_not(sym_conn_is_special))
        init_edges_special = tf.boolean_mask(init_edges, sym_conn_is_special)

        # Setup edge update function
        if use_edge_updates:
            edge_msg_fn = edge_update
            edges_normal = compute_messages(
                hidden_state,
                sym_conn_normal,
                init_edges_normal,
                edge_msg_fn,
                tf.identity,
                include_receiver=True,
                include_sender=True,
                only_messages=True,
            )
        else:
            edges_normal = init_edges_normal

        if use_edge_updates:
            edge_msg_fn = edge_update
            edges_special = compute_messages(
                hidden_state,
                sym_conn_special,
                init_edges_special,
                edge_msg_fn,
                tf.identity,
                include_receiver=True,
                include_sender=True,
                only_messages=True,
            )
        else:
            edges_special = init_edges_special

        # Setup interaction messages
        msg_fn = create_msg_function(hidden_state_len)
        act_fn = tf.identity
        for i in range(num_passes):
            if msg_share_weights:
                scope_suffix = ""
                reuse = i > 0
            else:
                scope_suffix = "%d" % i
                reuse = False
            with tf.variable_scope("msg_normal" + scope_suffix, reuse=reuse):
                sum_msg_normal = compute_messages(
                    hidden_state,
                    sym_conn_normal,
                    edges_normal,
                    msg_fn,
                    act_fn,
                    include_receiver=True,
                    mean_messages=avg_msg,
                )
            with tf.variable_scope("msg_special" + scope_suffix, reuse=reuse):
                sum_msg_special = compute_messages(
                    hidden_state,
                    sym_conn_special,
                    edges_special,
                    msg_fn,
                    act_fn,
                    include_receiver=True,
                    mean_messages=avg_msg,
                )
            with tf.variable_scope("update_normal" + scope_suffix, reuse=reuse):
                hidden_state += msgnet.defaults.mlp(
                    sum_msg_normal,
                    [hidden_state_len, hidden_state_len],
                    activation=msgnet.defaults.nonlinearity,
                    weights_initializer=msgnet.defaults.initializer,
                )
            with tf.variable_scope("update_special" + scope_suffix, reuse=reuse):
                hidden_state += msgnet.defaults.mlp(
                    sum_msg_special,
                    [hidden_state_len, hidden_state_len],
                    activation=msgnet.defaults.nonlinearity,
                    weights_initializer=msgnet.defaults.initializer,
                )
            with tf.variable_scope("edge_update_normal" + scope_suffix, reuse=reuse):
                if use_edge_updates and (i < (num_passes - 1)):
                    edges_normal = compute_messages(
                        hidden_state,
                        sym_conn_normal,
                        edges_normal,
                        edge_msg_fn,
                        tf.identity,
                        include_receiver=True,
                        include_sender=True,
                        only_messages=True,
                    )
            with tf.variable_scope("edge_update_special" + scope_suffix, reuse=reuse):
                if use_edge_updates and (i < (num_passes - 1)):
                    edges_special = compute_messages(
                        hidden_state,
                        sym_conn_special,
                        edges_special,
                        edge_msg_fn,
                        tf.identity,
                        include_receiver=True,
                        include_sender=True,
                        only_messages=True,
                    )

            nodes_out = tf.identity(hidden_state, name="nodes_out")

        # Setup readout function
        with tf.variable_scope("readout_edge"):
            if self.edge_output_fn is not None:
                self.edge_out = edge_output_fn(edges_normal)
        with tf.variable_scope("readout_graph"):
            if self.readout_fn is not None:
                graph_out = self.readout_fn(nodes_out, self.sym_segments)

        self.nodes_out = nodes_out
        self.graph_out_normalized = tf.identity(graph_out, name="graph_out_normalized")

        # Denormalize graph_out for making predictions on original scale
        if self.readout_fn.is_sum:
            mean = self.sym_target_mean * tf.expand_dims(
                tf.cast(self.sym_set_len, tf.float32), -1
            )
        else:
            mean = self.sym_target_mean
        self.graph_out = tf.add(graph_out * self.sym_target_std, mean, name="graph_out")

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=24, max_to_keep=3)

    def save(self, session, destination, global_step):
        self.saver.save(session, destination, global_step=global_step)

    def load(self, session, path):
        self.saver.restore(session, path)

    def get_nodes_out(self):
        return self.nodes_out

    def get_graph_out(self):
        if self.readout_fn is None:
            raise NotImplementedError("No readout function given")
        return self.graph_out

    def get_graph_out_normalized(self):
        if self.readout_fn is None:
            raise NotImplementedError("No readout function given")
        return self.graph_out_normalized

    def get_normalization(self):
        return self.sym_target_mean, self.sym_target_std

    def get_readout_function(self):
        return self.readout_fn

    def get_edges_out(self):
        if self.edge_output_fn is None:
            raise NotImplementedError("No edges output network given")
        return self.edge_out

    def get_input_symbols(self):
        return self.input_symbols
