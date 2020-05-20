from typing import List, Dict
import math
import ase
import torch
from torch import nn
import layer
from layer import ShiftedSoftplus


class DensityModel(nn.Module):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        gaussian_expansion_step=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.atom_model = AtomRepresentationModel(
            num_interactions, hidden_state_size, cutoff, gaussian_expansion_step,
        )

        self.probe_model = ProbeMessageModel(
            num_interactions, hidden_state_size, cutoff, gaussian_expansion_step,
        )

    def forward(self, input_dict):
        atom_representation = self.atom_model(input_dict)
        probe_result = self.probe_model(input_dict, atom_representation)
        return probe_result


class ProbeMessageModel(nn.Module):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        gaussian_expansion_step,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.cutoff = cutoff
        self.gaussian_expansion_step = gaussian_expansion_step

        edge_size = int(math.ceil(self.cutoff / self.gaussian_expansion_step))

        # Setup interaction networks
        self.messagesum_layers = nn.ModuleList(
            [
                layer.MessageSum(hidden_state_size, edge_size, self.cutoff, include_receiver=True)
                for _ in range(num_interactions)
            ]
        )

        # Setup transitions networks
        self.probe_state_gate_functions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_state_size, hidden_state_size),
                    ShiftedSoftplus(),
                    nn.Linear(hidden_state_size, hidden_state_size),
                    nn.Sigmoid(),
                )
                for _ in range(num_interactions)
            ]
        )
        self.probe_state_transition_functions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_state_size, hidden_state_size),
                    ShiftedSoftplus(),
                    nn.Linear(hidden_state_size, hidden_state_size),
                )
                for _ in range(num_interactions)
            ]
        )

        # Setup readout function
        self.readout_function = nn.Sequential(
            nn.Linear(hidden_state_size, hidden_state_size),
            ShiftedSoftplus(),
            nn.Linear(hidden_state_size, 1),
        )

    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
        atom_representation: List[torch.Tensor],
    ):
        # Unpad and concatenate edges and features into batch (0th) dimension
        edge_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_nodes"].device),
                    input_dict["num_nodes"][:-1],
                )
            ),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]

        # Unpad and concatenate probe edges into batch (0th) dimension
        probe_edges_features = layer.unpad_and_cat(
            input_dict["probe_edges_features"], input_dict["num_probe_edges"]
        )
        batch_size = input_dict["probe_edges"].shape[0]
        edge_probe_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_probes"].device),
                    input_dict["num_probes"][:-1],
                )
            ),
            dim=0,
        )
        edge_probe_offset = edge_probe_offset[:, None, None]
        edge_probe_offset = torch.cat((edge_offset, edge_probe_offset), dim=2)
        probe_edges = input_dict["probe_edges"] + edge_probe_offset
        probe_edges = layer.unpad_and_cat(probe_edges, input_dict["num_probe_edges"])

        # Expand edge features in Gaussian basis
        probe_edge_state = layer.gaussian_expansion(
            probe_edges_features, [(0.0, self.gaussian_expansion_step, self.cutoff)]
        )

        # Apply interaction layers
        probe_state = torch.zeros(
            (torch.sum(input_dict["num_probes"]), self.hidden_state_size),
            device=atom_representation[0].device,
        )
        for msg_layer, gate_layer, state_layer, nodes in zip(
            self.messagesum_layers,
            self.probe_state_gate_functions,
            self.probe_state_transition_functions,
            atom_representation,
        ):
            msgsum = msg_layer(
                nodes, probe_edges, probe_edge_state, probe_edges_features, probe_state,
            )
            gates = gate_layer(probe_state)
            probe_state = probe_state * gates + (1 - gates) * state_layer(msgsum)

        # Restack probe states
        probe_output = self.readout_function(probe_state).squeeze(1)
        probe_output = layer.pad_and_stack(
            torch.split(probe_output, list(input_dict["num_probes"].detach().cpu().numpy()), dim=0)
            #probe_output.reshape((-1, input_dict["num_probes"][0]))
        )
        return probe_output


class AtomRepresentationModel(nn.Module):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        gaussian_expansion_step,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.cutoff = cutoff
        self.gaussian_expansion_step = gaussian_expansion_step

        edge_size = int(math.ceil(self.cutoff / self.gaussian_expansion_step))

        # Setup interaction networks
        self.interactions = nn.ModuleList(
            [
                layer.Interaction(hidden_state_size, edge_size, self.cutoff, include_receiver=True)
                for _ in range(num_interactions)
            ]
        )

        # Atom embeddings
        self.atom_embeddings = nn.Embedding(
            len(ase.data.atomic_numbers), self.hidden_state_size
        )

    def forward(self, input_dict):
        # Unpad and concatenate edges and features into batch (0th) dimension
        edges_features = layer.unpad_and_cat(
            input_dict["atom_edges_features"], input_dict["num_atom_edges"]
        )
        edge_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_nodes"].device),
                    input_dict["num_nodes"][:-1],
                )
            ),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]
        edges = input_dict["atom_edges"] + edge_offset
        edges = layer.unpad_and_cat(edges, input_dict["num_atom_edges"])

        # Unpad and concatenate all nodes into batch (0th) dimension
        nodes = layer.unpad_and_cat(input_dict["nodes"], input_dict["num_nodes"])
        nodes = self.atom_embeddings(nodes)

        # Expand edge features in Gaussian basis
        edge_state = layer.gaussian_expansion(
            edges_features, [(0.0, self.gaussian_expansion_step, self.cutoff)]
        )

        nodes_list = []
        # Apply interaction layers
        for int_layer in self.interactions:
            nodes = int_layer(nodes, edges, edge_state, edges_features)
            nodes_list.append(nodes)

        return nodes_list
