from typing import Tuple, List
import itertools
import torch
from torch import nn
import numpy as np


def pad_and_stack(tensors: List[torch.Tensor]):
    """ Pad list of tensors if tensors are arrays and stack if they are scalars """
    if tensors[0].shape:
        return torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=0
        )
    return torch.stack(tensors)


def shifted_softplus(x):
    """
    Compute shifted soft-plus activation function.
    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)

    Args:
        x (torch.Tensor): input tensor.

    Returns:
        torch.Tensor: shifted soft-plus of input.

    """
    return nn.functional.softplus(x) - np.log(2.0)


class ShiftedSoftplus(nn.Module):
    def forward(self, x):
        return shifted_softplus(x)


def unpad_and_cat(stacked_seq: torch.Tensor, seq_len: torch.Tensor):
    """
    Unpad and concatenate by removing batch dimension

    Args:
        stacked_seq: (batch_size, max_length, *) Tensor
        seq_len: (batch_size) Tensor with length of each sequence

    Returns:
        (prod(seq_len), *) Tensor

    """
    unstacked = stacked_seq.unbind(0)
    unpadded = [
        torch.narrow(t, 0, 0, l) for (t, l) in zip(unstacked, seq_len.unbind(0))
    ]
    return torch.cat(unpadded, dim=0)


def sum_splits(values: torch.Tensor, splits: torch.Tensor):
    """
    Sum across dimension 0 of the tensor `values` in chunks
    defined in `splits`

    Args:
        values: Tensor of shape (`prod(splits)`, *)
        splits: 1-dimensional tensor with size of each chunk

    Returns:
        Tensor of shape (`splits.shape[0]`, *)

    """
    # prepare an index vector for summation
    ind = torch.zeros(splits.sum(), dtype=splits.dtype, device=splits.device)
    ind[torch.cumsum(splits, dim=0)[:-1]] = 1
    ind = torch.cumsum(ind, dim=0)
    # prepare the output
    sum_y = torch.zeros(
        splits.shape + values.shape[1:], dtype=values.dtype, device=values.device
    )
    # do the actual summation
    sum_y.index_add_(0, ind, values)
    return sum_y


def gaussian_expansion(input_x: torch.Tensor, expand_params: List[Tuple]):
    """
    Expand each feature in a number of Gaussian basis function.
    Expand_params is a list of length input_x.shape[1]

    Args:
        input_x: (num_edges, num_features) tensor
        expand_params: list of None or (start, step, stop) tuples

    Returns:
        (num_edges, ``ceil((stop - start)/step)``) tensor

    """
    feat_list = torch.unbind(input_x, dim=1)
    expanded_list = []
    for step_tuple, feat in itertools.zip_longest(expand_params, feat_list):
        assert feat is not None, "Too many expansion parameters given"
        if step_tuple:
            start, step, stop = step_tuple
            feat_expanded = torch.unsqueeze(feat, dim=1)
            sigma = step
            basis_mu = torch.arange(
                start, stop, step, device=input_x.device, dtype=input_x.dtype
            )
            expanded_list.append(
                torch.exp(-((feat_expanded - basis_mu) ** 2) / (2.0 * sigma ** 2))
            )
        else:
            expanded_list.append(torch.unsqueeze(feat, 1))
    return torch.cat(expanded_list, dim=1)


class SchnetMessageFunction(nn.Module):
    def __init__(self, input_size, edge_size, output_size, hard_cutoff):
        super().__init__()
        self.msg_function_edge = nn.Sequential(
            nn.Linear(edge_size, output_size),
            ShiftedSoftplus(),
            nn.Linear(output_size, output_size),
        )
        self.msg_function_node = nn.Sequential(
            nn.Linear(input_size, input_size),
            ShiftedSoftplus(),
            nn.Linear(input_size, output_size),
        )

        self.soft_cutoff_func = lambda x: 1.0 - torch.sigmoid(
            5 * (x - (hard_cutoff - 1.5))
        )

    def forward(self, node_state, edge_state, edge_distance):
        gates = self.msg_function_edge(edge_state) * self.soft_cutoff_func(
            edge_distance
        )
        nodes = self.msg_function_node(node_state)
        return nodes * gates


class Interaction(nn.Module):
    def __init__(self, node_size, edge_size, cutoff, include_receiver=False):
        super().__init__()

        self.message_sum_module = MessageSum(
            node_size, edge_size, cutoff, include_receiver
        )

        self.state_transition_function = nn.Sequential(
            nn.Linear(node_size, node_size),
            ShiftedSoftplus(),
            nn.Linear(node_size, node_size),
        )

    def forward(self, node_state, edges, edge_state, edges_distance):

        # Compute sum of messages
        message_sum = self.message_sum_module(
            node_state, edges, edge_state, edges_distance
        )

        # State transition
        new_state = node_state + self.state_transition_function(message_sum)

        return new_state


class MessageSum(nn.Module):
    def __init__(self, node_size, edge_size, cutoff, include_receiver):
        super().__init__()

        self.include_receiver = include_receiver

        if include_receiver:
            input_size = node_size * 2
        else:
            input_size = node_size

        self.message_function = SchnetMessageFunction(input_size, edge_size, node_size, cutoff)

    def forward(
        self, node_state, edges, edge_state, edges_distance, receiver_nodes=None
    ):
        """

        Args:
            node_state: [num_nodes, n_node_features] State of input nodes
            edges: [num_edges, 2] array of sender and receiver indices
            edge_state: [num_edges, n_features] array of edge features
            edges_distance: [num_edges, 1] array of distances
            receiver_nodes: If given, use these nodes as receiver nodes instead of node_state

        Returns:
            sum of messages to each node

        """
        # Compute all messages
        if self.include_receiver:
            if receiver_nodes is not None:
                senders = node_state[edges[:, 0]]
                receivers = receiver_nodes[edges[:, 1]]
                nodes = torch.cat((senders, receivers), dim=1)
            else:
                num_edges = edges.shape[0]
                nodes = torch.reshape(node_state[edges], (num_edges, -1))
        else:
            nodes = node_state[edges[:, 0]]  # Only include sender in messages
        messages = self.message_function(nodes, edge_state, edges_distance)

        # Sum messages
        if receiver_nodes is not None:
            message_sum = torch.zeros_like(receiver_nodes)
        else:
            message_sum = torch.zeros_like(node_state)
        message_sum.index_add_(0, edges[:, 1], messages)

        return message_sum


class EdgeUpdate(nn.Module):
    def __init__(self, edge_size, node_size):
        super().__init__()

        self.node_size = node_size
        self.edge_update_mlp = nn.Sequential(
            nn.Linear(2 * node_size + edge_size, 2 * edge_size),
            ShiftedSoftplus(),
            nn.Linear(2 * edge_size, edge_size),
        )

    def forward(self, edge_state, edges, node_state):
        combined = torch.cat(
            (node_state[edges].view(-1, 2 * self.node_size), edge_state), axis=1
        )
        return self.edge_update_mlp(combined)
