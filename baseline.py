import argparse
import datetime
import json
import os
import random

import torch
import numpy as np

# Importing necessary modules from the project
from models.model import init_nets
from utils.utils import get_dataloader
from utils.trainer import fit, test

# Function to parse command line arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="twitter", help="dataset used for training")
    parser.add_argument("--batch_size", type=int, default=64, help="total sum of input batch size for training (default: 128)")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate (default: 0.001)")
    parser.add_argument("--epochs", type=int, default=10, help="number of local epochs")
    parser.add_argument("--optimizer", type=str, default="adam", help="the optimizer")
    parser.add_argument("--init_seed", type=int, default=0, help="Random seed")
    parser.add_argument("--datadir", type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument("--logdir", type=str, required=False, default="./logs/", help="Log directory path")
    parser.add_argument("--modeldir", type=str, required=False, default="./models/", help="Model directory path")
    parser.add_argument("--device", type=str, default="cuda:0", help="The device to run the program")
    parser.add_argument("--vocab", type=str, default="vocab.json", help="vocab file")
    parser.add_argument("--vocab_size", type=int, default=13354, help="size of the vocabulary")
    parser.add_argument("--hidden_dim", type=int, default=128, help="dimension of hidden layer")

    args = parser.parse_args()
    return args

# Function to set the seed for all random number generators to ensure reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    # Get arguments
    args = get_args()

    out_dim=2
    vocab_size=args.vocab_size

    # Create directory to save log
    argument_path = f"{args.dataset}_arguments-%s.json" % datetime.datetime.now().strftime(
        "%Y-%m-%d-%H%M-%S"
    )

    # Save arguments
    with open(os.path.join(args.logdir, argument_path), "w") as f:
        json.dump(str(args), f)

    device = torch.device(args.device)

    # Set seed
    set_seed(args.init_seed)

    # Get dataloader
    (train_dl, test_dl, train_ds, test_ds) = get_dataloader(
        args.datadir, args.batch_size, args
    )

    print("len train_dl_global:", len(train_ds))
    data_size = len(test_ds)

    # Initializing net
    print("Initializing nets")
    models = init_nets(out_dim, args.vocab, vocab_size, 1, args)

    model = models[0]

    # Train the model
    fit(
        model,
        train_dl,
        args.epochs,
        args.lr,
        args.optimizer,
        device=args.device,
    )

    # Test the model
    test(
        model,
        test_dl,
        args,
        device=args.device
    )

    # Save the model
    torch.save(model.state_dict(), args.modeldir + "basemodel_" + args.dataset + ".pth")