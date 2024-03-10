import argparse
import copy
import datetime
import json
import os
import random

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from models.model import init_nets
from utils.utils import get_dataloader, partition_data
from utils.trainer import get_accuracy_with_softmax, get_accuracy_without_softmax


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="twitter", help="dataset used for training")
    parser.add_argument("--partition", type=str, default="noniid", help="the data partitioning strategy")
    parser.add_argument("--batch_size", type=int, default=64, help="total sum of input batch size for training (default: 128)")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate (default: 0.001)")
    parser.add_argument("--epochs", type=int, default=5, help="number of local epochs")
    parser.add_argument("--n_parties", type=int, default=10, help="number of workers in a distributed cluster")
    parser.add_argument("--comm_round", type=int, default=10, help="number of maximum communication roun")
    parser.add_argument("--init_seed", type=int, default=0, help="Random seed")
    parser.add_argument("--traindir", type=str, required=False, default="./data/train.csv", help="Data directory")
    parser.add_argument("--testdir", type=str, required=False, default="./data/test.csv", help="Data directory")
    parser.add_argument("--logdir", type=str, required=False, default="./logs/", help="Log directory path")
    parser.add_argument("--modeldir", type=str, required=False, default="./models/", help="Model directory path")
    parser.add_argument("--beta", type=float, default=0.5, help="The parameter for the dirichlet distribution for data partitioning")
    parser.add_argument("--sample_fraction", type=float, default=1.0, help="how many clients are sampled in each round")
    parser.add_argument("--device", type=str, default="cuda:0", help="The device to run the program")
    parser.add_argument("--optimizer", type=str, default="adam", help="the optimizer")
    parser.add_argument("--vocab", type=str, default="vocab.json", help="vocab file")
    parser.add_argument("--hidden_dim", type=int, default=128, help="dimension of hidden layer")

    args = parser.parse_args()
    return args


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

def fit(
    net,
    train_dataloader,
    epochs,
    lr,
    args_optimizer,
    apply_softmax=False,
    device="cpu",
):
    net.cuda()

    # Set optimizer
    if args_optimizer == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=1e-5)
    elif args_optimizer == "amsgrad":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=1e-5, amsgrad=True
        )
    elif args_optimizer == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9, weight_decay=1e-5
        )

    criterion = nn.CrossEntropyLoss()
    net.train()

    for epoch in range(epochs):
        print()
        train_running_loss = 0
        train_running_acc = 0   
        
        # tqdm_epoch_iterator.set_description(desc=f"epochs {epoch+1}/{n_epochs}")
        tqdm_train_iterator = tqdm(enumerate(train_dataloader),
                                    desc=f"[train]{epoch+1}/{epochs}",
                                    ascii=True,leave=True,
                                    total=len(train_dataloader),
                                    colour="green",position=0)
                    
        for batch_idx,(data,target) in tqdm_train_iterator:

            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            y_pred = net(data)

            loss = criterion(y_pred,target)
            loss.backward()
            optimizer.step()
            
            train_running_loss += loss.item()
            
            if apply_softmax:
                train_running_acc += get_accuracy_with_softmax(y_pred.detach(),target)
            else:
                train_running_acc += get_accuracy_without_softmax(y_pred.detach(),target)
                
            tqdm_train_iterator.set_postfix(avg_train_acc=f"{train_running_acc/(batch_idx+1):0.4f}",
                                            avg_train_loss=f"{(train_running_loss/(batch_idx+1)):0.4f}")

    net.eval()
    print(" ** Training complete **")


def local_train_net(
    nets,
    args,
    net_dataidx_map,
    train_dl_local_dict,
    device="cpu",
):

    n_epoch = args.epochs
    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]
        print("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        train_dl_local = train_dl_local_dict[net_id]
        fit(
            net,
            train_dl_local,
            n_epoch,
            args.lr,
            args.optimizer,
            device=args.device,
        )

    return nets


if __name__ == "__main__":
    vocab_size=195675
    embedding_dim=768
    out_dim=2

    args = get_args()

    # Create directory to save log and model
    argument_path = f"{args.dataset}-{args.batch_size}-{args.n_parties}_arguments-%s.json" % datetime.datetime.now().strftime(
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
        args.dataset, args.traindir, args.testdir, args.logdir, args.partition, args.n_parties, beta=args.beta
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
        args.dataset, args.vocab, args.traindir, args.testdir, args.batch_size, args.batch_size * 2
    )

    print("len train_dl_global:", len(train_ds_global))
    train_dl = None
    data_size = len(test_ds)

    # Initializing net from each local party.
    print("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(embedding_dim, out_dim, vocab_size, args.n_parties, args, device="cpu")
    global_models, global_model_meta_data, global_layer_type = init_nets(embedding_dim, out_dim, vocab_size, 1, args, device="cpu")

    global_model = global_models[0]
    n_comm_rounds = args.comm_round

    train_dl_local_dict = {}
    net_id = 0

    # Distribute dataset and dataloader to each local party
    # We use two dataloaders for training FedX (train_dataloader, random_dataloader), 
    # and their batch sizes (args.batch_size // 2) are summed up to args.batch_size
    for net in nets:
        dataidxs = net_dataidx_map[net_id]
        train_dl_local, _, _, _ = get_dataloader(
            args.dataset, args.vocab, args.traindir, args.testdir, args.batch_size, args.batch_size * 2, dataidxs
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
            device=device,
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
        # test_acc_1, test_acc_5 = test(global_model, val_dl_global, test_dl)
        # print(">> Global Model Test accuracy Top1: %f" % test_acc_1)
        # print(">> Global Model Test accuracy Top5: %f" % test_acc_5)

    # Save the final round's local and global models
    # torch.save(global_model.state_dict(), args.modeldir + "globalmodel" + args.log_file_name + ".pth")
    # torch.save(nets[0].state_dict(), args.modeldir + "localmodel0" + args.log_file_name + ".pth")
