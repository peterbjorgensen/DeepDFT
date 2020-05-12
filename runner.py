import os
import sys
import json
import argparse
import math
import logging
import itertools

import numpy as np
import torch

import densitymodel
import dataset


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Train graph convolution network", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="Load model parameters from previous run",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=5.0,
        help="Atomic interaction cutoff distance [Å]",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help="Train/test/validation split file json",
    )
    parser.add_argument(
        "--num_interactions",
        type=int,
        default=3,
        help="Number of interaction layers used",
    )
    parser.add_argument(
        "--node_size", type=int, default=64, help="Size of hidden node states"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/model_output",
        help="Path to output directory",
    )
    parser.add_argument(
        "--dataset", type=str, default="data/qm9.db", help="Path to ASE database",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=int(1e6),
        help="Maximum number of optimisation steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Set which device to use for training e.g. 'cuda' or 'cpu'",
    )

    return parser.parse_args(arg_list)


def split_data(dataset, args):
    # Load or generate splits
    if args.split_file:
        with open(args.split_file, "r") as fp:
            splits = json.load(fp)
    else:
        datalen = len(dataset)
        num_validation = int(math.ceil(datalen * 0.10))
        indices = np.random.permutation(len(dataset))
        splits = {
            "train": indices[num_validation:].tolist(),
            "validation": indices[:num_validation].tolist(),
        }

    # Save split file
    with open(os.path.join(args.output_dir, "datasplits.json"), "w") as f:
        json.dump(splits, f)

    # Split the dataset
    datasplits = {}
    for key, indices in splits.items():
        datasplits[key] = torch.utils.data.Subset(dataset, indices)
    return datasplits


def eval_model(model, dataloader, device):
    running_ae = 0
    running_se = 0
    running_count = 0
    for batch in dataloader:
        device_batch = {
            k: v.to(device=device, non_blocking=True) for k, v in batch.items()
        }
        with torch.no_grad():
            outputs = model(device_batch).detach().cpu().numpy()
        targets = batch["probe_target"].detach().cpu().numpy()

        running_ae += np.sum(np.abs(targets - outputs))
        running_se += np.sum(np.square(targets - outputs))
        running_count += torch.prod(batch["num_probes"]).detach().cpu().numpy()

    mae = running_ae / running_count
    rmse = np.sqrt(running_se / running_count)

    return mae, rmse


def get_normalization(dataset, per_atom=True):
    try:
        num_targets = len(dataset.transformer.targets)
    except AttributeError:
        num_targets = 1
    x_sum = torch.zeros(num_targets)
    x_2 = torch.zeros(num_targets)
    num_objects = 0
    for sample in dataset:
        x = sample["targets"]
        if per_atom:
            x = x / sample["num_nodes"]
        x_sum += x
        x_2 += x ** 2.0
        num_objects += 1
    # Var(X) = E[X^2] - E[X]^2
    x_mean = x_sum / num_objects
    x_var = x_2 / num_objects - x_mean ** 2.0

    return x_mean, torch.sqrt(x_var)


def main():
    args = get_arguments()

    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(args.output_dir, "printlog.txt"), mode="w"
            ),
            logging.StreamHandler(),
        ],
    )

    # Save command line args
    with open(os.path.join(args.output_dir, "commandline_args.txt"), "w") as f:
        f.write("\n".join(sys.argv[1:]))
    # Save parsed command line arguments
    with open(os.path.join(args.output_dir, "arguments.json"), "w") as f:
        json.dump(vars(args), f)

    # Setup dataset and loader
    logging.info("loading data %s", args.dataset)
    densitydata = dataset.DensityData(args.dataset,)
    densitydata = dataset.BufferData(densitydata)  # Load data into host memory

    # Split data into train and validation sets
    datasplits = split_data(densitydata, args)

    # Setup loaders
    train_loader = torch.utils.data.DataLoader(
        datasplits["train"],
        2,
        num_workers=0,
        sampler=torch.utils.data.RandomSampler(datasplits["train"]),
        collate_fn=dataset.CollateFuncRandomSample(args.cutoff, 100, pin_memory=True),
    )
    val_loader = torch.utils.data.DataLoader(
        datasplits["validation"],
        32,
        collate_fn=dataset.CollateFuncRandomSample(args.cutoff, 100, pin_memory=False),
        num_workers=0,
    )

    # Initialise model
    device = torch.device(args.device)
    net = densitymodel.DensityModel(args.num_interactions, args.node_size, args.cutoff,)
    net = net.to(device)

    # Setup optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()
    scheduler_fn = lambda step: 0.96 ** (step / 100000)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_fn)

    log_interval = 10000
    running_loss = 0
    running_loss_count = 0
    best_val_mae = np.inf
    step = 0
    # Restore checkpoint
    if args.load_model:
        state_dict = torch.load(args.load_model)
        net.load_state_dict(state_dict["model"])
        step = state_dict["step"]
        best_val_mae = state_dict["best_val_mae"]
        optimizer.load_state_dict(state_dict["optimizer"])
        scheduler.load_state_dict(state_dict["scheduler"])

    logging.info("start training")
    for epoch in itertools.count():
        for batch_host in train_loader:
            # Transfer to 'device'
            batch = {
                k: v.to(device=device, non_blocking=True)
                for (k, v) in batch_host.items()
            }

            # Reset gradient
            optimizer.zero_grad()

            # Forward, backward and optimize
            outputs = net(batch)
            loss = criterion(outputs, batch["probe_target"])
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            running_loss += loss_value * batch["probe_target"].shape[0]
            running_loss_count += batch["probe_target"].shape[0]

            # print(step, loss_value)
            # Validate and save model
            if (step % log_interval == 0) or ((step + 1) == args.max_steps):
                train_loss = running_loss / running_loss_count
                running_loss = running_loss_count = 0

                val_mae, val_rmse = eval_model(net, val_loader, device)

                logging.info(
                    "step=%d, val_mae=%g, val_rmse=%g, sqrt(train_loss)=%g",
                    step,
                    val_mae,
                    val_rmse,
                    math.sqrt(train_loss),
                )

                # Save checkpoint
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    torch.save(
                        {
                            "model": net.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "step": step,
                            "best_val_mae": best_val_mae,
                        },
                        os.path.join(args.output_dir, "best_model.pth"),
                    )
            step += 1

            scheduler.step()

            if step >= args.max_steps:
                logging.info("Max steps reached, exiting")
                sys.exit(0)


if __name__ == "__main__":
    main()
