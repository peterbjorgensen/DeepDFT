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
            num_interactions,
            hidden_state_size,
            cutoff,
            gaussian_expansion_step,
        )

        self.probe_model = ProbeMessageModel(
            num_interactions,
            hidden_state_size,
            cutoff,
            gaussian_expansion_step,
        )

    def forward(self, input_dict):
        atom_representation = self.atom_model(input_dict)
        probe_result = self.probe_model(input_dict, atom_representation)
        return probe_result

class PainnDensityModel(nn.Module):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        distance_embedding_size=30,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.atom_model = PainnAtomRepresentationModel(
            num_interactions,
            hidden_state_size,
            cutoff,
            distance_embedding_size,
        )

        self.probe_model = PainnProbeMessageModel(
            num_interactions,
            hidden_state_size,
            cutoff,
            distance_embedding_size,
        )

    def forward(self, input_dict):
        atom_representation_scalar, atom_representation_vector = self.atom_model(input_dict)
        probe_result = self.probe_model(input_dict, atom_representation_scalar, atom_representation_vector)
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
                layer.MessageSum(
                    hidden_state_size, edge_size, self.cutoff, include_receiver=True
                )
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
        compute_iri=False,
        compute_dori=False,
        compute_hessian=False,
    ):
        if compute_iri or compute_dori or compute_hessian:
            input_dict["probe_xyz"].requires_grad_()

        # Unpad and concatenate edges and features into batch (0th) dimension
        atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        probe_xyz = layer.unpad_and_cat(
            input_dict["probe_xyz"], input_dict["num_probes"]
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

        # Unpad and concatenate probe edges into batch (0th) dimension
        probe_edges_displacement = layer.unpad_and_cat(
            input_dict["probe_edges_displacement"], input_dict["num_probe_edges"]
        )
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

        # Compute edge distances
        probe_edges_features = layer.calc_distance_to_probe(
            atom_xyz,
            probe_xyz,
            input_dict["cell"],
            probe_edges,
            probe_edges_displacement,
            input_dict["num_probe_edges"],
        )

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
                nodes,
                probe_edges,
                probe_edge_state,
                probe_edges_features,
                probe_state,
            )
            gates = gate_layer(probe_state)
            probe_state = probe_state * gates + (1 - gates) * state_layer(msgsum)

        # Restack probe states
        probe_output = self.readout_function(probe_state).squeeze(1)
        probe_output = layer.pad_and_stack(
            torch.split(
                probe_output,
                list(input_dict["num_probes"].detach().cpu().numpy()),
                dim=0,
            )
            # torch.split(probe_output, input_dict["num_probes"], dim=0)
            # probe_output.reshape((-1, input_dict["num_probes"][0]))
        )

        if compute_iri or compute_dori or compute_hessian:
            dp_dxyz = torch.autograd.grad(
                probe_output,
                input_dict["probe_xyz"],
                grad_outputs=torch.ones_like(probe_output),
                retain_graph=True,
                create_graph=True,
            )[0]

        grad_probe_outputs = {}

        if compute_iri:
            iri = torch.linalg.norm(dp_dxyz, dim=2)/(torch.pow(probe_output, 1.1))
            grad_probe_outputs["iri"] = iri

        if compute_dori:
            ##
            ## DORI(r) = phi(r) / (1 + phi(r))
            ## phi(r) = ||grad(||grad(rho(r))/rho||^2)||^2 / ||grad(rho(r))/rho(r)||^6
            ##
            norm_grad_2 = torch.linalg.norm(dp_dxyz/torch.unsqueeze(probe_output, 2), dim=2)**2

            grad_norm_grad_2 = torch.autograd.grad(
                norm_grad_2,
                input_dict["probe_xyz"],
                grad_outputs=torch.ones_like(norm_grad_2),
                only_inputs=True,
                retain_graph=True,
                create_graph=True,
            )[0].detach()

            phi_r = torch.linalg.norm(grad_norm_grad_2, dim=2)**2 / (norm_grad_2**3)

            dori = phi_r / (1 + phi_r)
            grad_probe_outputs["dori"] = dori

        if compute_hessian:
            hessian_shape = (input_dict["probe_xyz"].shape[0], input_dict["probe_xyz"].shape[1], 3, 3)
            hessian = torch.zeros(hessian_shape, device=probe_xyz.device, dtype=probe_xyz.dtype)
            for dim_idx, grad_out in enumerate(torch.unbind(dp_dxyz, dim=-1)):
                dp2_dxyz2 = torch.autograd.grad(
                    grad_out,
                    input_dict["probe_xyz"],
                    grad_outputs=torch.ones_like(grad_out),
                    only_inputs=True,
                    retain_graph=True,
                    create_graph=True,
                )[0]
                hessian[:, :, dim_idx] = dp2_dxyz2
            grad_probe_outputs["hessian"] = hessian


        if grad_probe_outputs:
            return probe_output, grad_probe_outputs
        else:
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
                layer.Interaction(
                    hidden_state_size, edge_size, self.cutoff, include_receiver=True
                )
                for _ in range(num_interactions)
            ]
        )

        # Atom embeddings
        self.atom_embeddings = nn.Embedding(
            len(ase.data.atomic_numbers), self.hidden_state_size
        )

    def forward(self, input_dict):
        # Unpad and concatenate edges and features into batch (0th) dimension
        edges_displacement = layer.unpad_and_cat(
            input_dict["atom_edges_displacement"], input_dict["num_atom_edges"]
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
        atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        nodes = layer.unpad_and_cat(input_dict["nodes"], input_dict["num_nodes"])
        nodes = self.atom_embeddings(nodes)

        # Compute edge distances
        edges_features = layer.calc_distance(
            atom_xyz,
            input_dict["cell"],
            edges,
            edges_displacement,
            input_dict["num_atom_edges"],
        )

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


class PainnAtomRepresentationModel(nn.Module):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        distance_embedding_size,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.cutoff = cutoff
        self.distance_embedding_size = distance_embedding_size

        # Setup interaction networks
        self.interactions = nn.ModuleList(
            [
                layer.PaiNNInteraction(
                    hidden_state_size, self.distance_embedding_size, self.cutoff
                )
                for _ in range(num_interactions)
            ]
        )
        self.scalar_vector_update = nn.ModuleList(
            [layer.PaiNNUpdate(hidden_state_size) for _ in range(num_interactions)]
        )

        # Atom embeddings
        self.atom_embeddings = nn.Embedding(
            len(ase.data.atomic_numbers), self.hidden_state_size
        )

    def forward(self, input_dict):
        # Unpad and concatenate edges and features into batch (0th) dimension
        edges_displacement = layer.unpad_and_cat(
            input_dict["atom_edges_displacement"], input_dict["num_atom_edges"]
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
        atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        nodes_scalar = layer.unpad_and_cat(input_dict["nodes"], input_dict["num_nodes"])
        nodes_scalar = self.atom_embeddings(nodes_scalar)
        nodes_vector = torch.zeros(
            (nodes_scalar.shape[0], 3, self.hidden_state_size),
            dtype=nodes_scalar.dtype,
            device=nodes_scalar.device,
        )

        # Compute edge distances
        edges_distance, edges_diff = layer.calc_distance(
            atom_xyz,
            input_dict["cell"],
            edges,
            edges_displacement,
            input_dict["num_atom_edges"],
            return_diff=True,
        )

        # Expand edge features in sinc basis
        edge_state = layer.sinc_expansion(
            edges_distance, [(self.distance_embedding_size, self.cutoff)]
        )

        nodes_list_scalar = []
        nodes_list_vector = []
        # Apply interaction layers
        for int_layer, update_layer in zip(
            self.interactions, self.scalar_vector_update
        ):
            nodes_scalar, nodes_vector = int_layer(
                nodes_scalar,
                nodes_vector,
                edge_state,
                edges_diff,
                edges_distance,
                edges,
            )
            nodes_scalar, nodes_vector = update_layer(nodes_scalar, nodes_vector)
            nodes_list_scalar.append(nodes_scalar)
            nodes_list_vector.append(nodes_vector)

        return nodes_list_scalar, nodes_list_vector


class PainnProbeMessageModel(nn.Module):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        distance_embedding_size,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.cutoff = cutoff
        self.distance_embedding_size = distance_embedding_size

        # Setup interaction networks
        self.message_layers = nn.ModuleList(
            [
                layer.PaiNNInteractionOneWay(
                    hidden_state_size, self.distance_embedding_size, self.cutoff
                )
                for _ in range(num_interactions)
            ]
        )
        self.scalar_vector_update = nn.ModuleList(
            [layer.PaiNNUpdate(hidden_state_size) for _ in range(num_interactions)]
        )

        # Setup readout function
        self.readout_function = nn.Sequential(
            nn.Linear(hidden_state_size, hidden_state_size),
            nn.SiLU(),
            nn.Linear(hidden_state_size, 1),
        )

    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
        atom_representation_scalar: List[torch.Tensor],
        atom_representation_vector: List[torch.Tensor],
        compute_iri=False,
        compute_dori=False,
        compute_hessian=False,
    ):
        if compute_iri or compute_dori or compute_hessian:
            input_dict["probe_xyz"].requires_grad_()

        # Unpad and concatenate edges and features into batch (0th) dimension
        atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        probe_xyz = layer.unpad_and_cat(
            input_dict["probe_xyz"], input_dict["num_probes"]
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

        # Unpad and concatenate probe edges into batch (0th) dimension
        probe_edges_displacement = layer.unpad_and_cat(
            input_dict["probe_edges_displacement"], input_dict["num_probe_edges"]
        )
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

        # Compute edge distances
        probe_edges_distance, probe_edges_diff = layer.calc_distance_to_probe(
            atom_xyz,
            probe_xyz,
            input_dict["cell"],
            probe_edges,
            probe_edges_displacement,
            input_dict["num_probe_edges"],
            return_diff=True,
        )

        # Expand edge features in sinc basis
        edge_state = layer.sinc_expansion(
            probe_edges_distance, [(self.distance_embedding_size, self.cutoff)]
        )

        # Apply interaction layers
        probe_state_scalar = torch.zeros(
            (torch.sum(input_dict["num_probes"]), self.hidden_state_size),
            device=atom_representation_scalar[0].device,
        )
        probe_state_vector = torch.zeros(
            (torch.sum(input_dict["num_probes"]), 3, self.hidden_state_size),
            device=atom_representation_scalar[0].device,
        )

        for msg_layer, update_layer, atom_nodes_scalar, atom_nodes_vector in zip(
            self.message_layers,
            self.scalar_vector_update,
            atom_representation_scalar,
            atom_representation_vector,
        ):
            probe_state_scalar, probe_state_vector = msg_layer(
                atom_nodes_scalar,
                atom_nodes_vector,
                probe_state_scalar,
                probe_state_vector,
                edge_state,
                probe_edges_diff,
                probe_edges_distance,
                probe_edges,
            )
            probe_state_scalar, probe_state_vector = update_layer(
                probe_state_scalar, probe_state_vector
            )

        # Restack probe states
        probe_output = self.readout_function(probe_state_scalar).squeeze(1)
        probe_output = layer.pad_and_stack(
            torch.split(
                probe_output,
                list(input_dict["num_probes"].detach().cpu().numpy()),
                dim=0,
            )
            # torch.split(probe_output, input_dict["num_probes"], dim=0)
            # probe_output.reshape((-1, input_dict["num_probes"][0]))
        )

        if compute_iri or compute_dori or compute_hessian:
            dp_dxyz = torch.autograd.grad(
                probe_output,
                input_dict["probe_xyz"],
                grad_outputs=torch.ones_like(probe_output),
                retain_graph=True,
                create_graph=True,
            )[0]

        grad_probe_outputs = {}

        if compute_iri:
            iri = torch.linalg.norm(dp_dxyz, dim=2)/(torch.pow(probe_output, 1.1))
            grad_probe_outputs["iri"] = iri

        if compute_dori:
            ##
            ## DORI(r) = phi(r) / (1 + phi(r))
            ## phi(r) = ||grad(||grad(rho(r))/rho||^2)||^2 / ||grad(rho(r))/rho(r)||^6
            ##
            norm_grad_2 = torch.linalg.norm(dp_dxyz/(torch.unsqueeze(probe_output, 2)), dim=2)**2

            grad_norm_grad_2 = torch.autograd.grad(
                norm_grad_2,
                input_dict["probe_xyz"],
                grad_outputs=torch.ones_like(norm_grad_2),
                only_inputs=True,
                retain_graph=True,
                create_graph=True,
            )[0].detach()

            phi_r = torch.linalg.norm(grad_norm_grad_2, dim=2)**2 / (norm_grad_2**3)

            dori = phi_r / (1 + phi_r)
            grad_probe_outputs["dori"] = dori

        if compute_hessian:
            hessian_shape = (input_dict["probe_xyz"].shape[0], input_dict["probe_xyz"].shape[1], 3, 3)
            hessian = torch.zeros(hessian_shape, device=probe_xyz.device, dtype=probe_xyz.dtype)
            for dim_idx, grad_out in enumerate(torch.unbind(dp_dxyz, dim=-1)):
                dp2_dxyz2 = torch.autograd.grad(
                    grad_out,
                    input_dict["probe_xyz"],
                    grad_outputs=torch.ones_like(grad_out),
                    only_inputs=True,
                    retain_graph=True,
                    create_graph=True,
                )[0]
                hessian[:, :, dim_idx] = dp2_dxyz2
            grad_probe_outputs["hessian"] = hessian


        if grad_probe_outputs:
            return probe_output, grad_probe_outputs
        else:
            return probe_output
