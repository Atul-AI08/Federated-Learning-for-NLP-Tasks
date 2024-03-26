import argparse
import copy
import datetime
import json
import os
import random

import torch
import numpy as np

# Importing necessary modules from the project
from models.model import init_nets
from utils.utils import get_dataloader, partition_data
from utils.trainer import fit, test

# Function to parse command line arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="twitter", help="dataset used for training")
    parser.add_argument("--partition", type=str, default="iid", help="Data partitioning strategy. Default is 'iid'.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training. Default is 64.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer. Default is 0.001.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of local epochs to run. Default is 3.")
    parser.add_argument("--n_parties", type=int, default=5, help="Number of workers in a distributed cluster. Default is 5.")
    parser.add_argument("--comm_round", type=int, default=5, help="Maximum number of communication rounds. Default is 5.")
    parser.add_argument("--init_seed", type=int, default=0, help="Seed for random number generation. Default is 0.")
    parser.add_argument("--datadir", type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument("--logdir", type=str, required=False, default="./logs/", help="Log directory path")
    parser.add_argument("--modeldir", type=str, required=False, default="./models/", help="Model directory path")
    parser.add_argument("--beta", type=float, default=0.6, help="Parameter for the Dirichlet distribution for data partitioning. Default is 0.6.")
    parser.add_argument("--sample_fraction", type=float, default=1.0, help="Fraction of clients to be sampled in each round. Default is 1.0.")
    parser.add_argument("--device", type=str, default="cuda:0", help="The device to run the program")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer to use for training. Default is 'adam'.")
    parser.add_argument("--vocab", type=str, default="vocab.json", help="File for the vocabulary.")
    parser.add_argument("--vocab_size", type=int, default=13354, help="Size of the vocabulary")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Dimension of the hidden layer in the neural network. Default is 128.")

    args = parser.parse_args()
    return args

# Function to set the seed for all random number generators to ensure reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

# Function to train the network locally
def local_train_net(
    nets,
    args,
    net_dataidx_map,
    train_dl_local_dict,
):

    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]
        print("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        train_dl_local = train_dl_local_dict[net_id]
        fit(
            net,
            train_dl_local,
            args.epochs,
            args.lr,
            args.optimizer,
            device=args.device,
        )

    return nets


if __name__ == "__main__":
    # Get command line arguments
    args = get_args()

    vocab_size=args.device
    out_dim=2

    # Create directory to save log and model
    argument_path = f"{args.dataset}-{args.n_parties}_arguments-%s.json" % datetime.datetime.now().strftime(
        "%Y-%m-%d-%H%M-%S"
    )

    # Save arguments
    with open(os.path.join(args.logdir, argument_path), "w") as f:
        json.dump(str(args), f)

    device = torch.device(args.device)

    # Set seed
    set_seed(args.init_seed)

    # Data partitioning with respect to the number of parties
    print("Partitioning data")
    (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts) = partition_data(
        args.dataset, args.datadir, args.partition, args.n_parties, beta=args.beta
    )

    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []

    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    n_classes = len(np.unique(y_train))

    # Get global dataloader (only used for evaluation)
    (train_dl_global, test_dl, train_ds_global, test_ds) = get_dataloader(
        args.datadir, args.batch_size, args
    )

    print("len train_dl_global:", len(train_ds_global))
    train_dl = None
    data_size = len(test_ds)

    # Initializing net from each local party.
    print("Initializing nets")
    nets = init_nets(out_dim, args.vocab, args.vocab_size, args.n_parties, args)
    global_models = init_nets(out_dim, args.vocab, args.vocab_size, 1, args)

    global_model = global_models[0]
    n_comm_rounds = args.comm_round

    train_dl_local_dict = {}
    net_id = 0

    # Distribute dataset and dataloader to each local party
    for net in nets:
        dataidxs = net_dataidx_map[net_id]
        train_dl_local, _, _, _ = get_dataloader(
            args.datadir, args.batch_size, args, dataidxs
        )
        train_dl_local_dict[net_id] = train_dl_local
        net_id += 1

    # Main training communication loop.
    for round in range(n_comm_rounds):
        print("in comm round:" + str(round))
        party_list_this_round = party_list_rounds[round]

        # Download global model from (virtual) central server
        global_w = global_model.state_dict()
        nets_this_round = {k: nets[k] for k in party_list_this_round}
        for net in nets_this_round.values():
            net.load_state_dict(global_w)

        # Train local model with local data
        local_train_net(
            nets_this_round,
            args,
            net_dataidx_map,
            train_dl_local_dict,
        )

        total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_parties)])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_parties)]

        # Averaging the local models' parameters to get global model
        for net_id, net in enumerate(nets_this_round.values()):
            net_para = net.state_dict()
            if net_id == 0:
                for key in net_para:
                    global_w[key] = net_para[key] * fed_avg_freqs[net_id]
            else:
                for key in net_para: 
                    global_w[key] += net_para[key] * fed_avg_freqs[net_id]

        global_model.load_state_dict(copy.deepcopy(global_w))

        # Evaluating the global model
        test(global_model, test_dl, args, verbose=False, device=args.device)

    # Final evaluation of the global model
    test(global_model, test_dl, args, device=args.device)

    # Save the final round's local and global models
    torch.save(global_model.state_dict(), args.modeldir + "globalmodel_" + args.dataset + args.partition + ".pth")