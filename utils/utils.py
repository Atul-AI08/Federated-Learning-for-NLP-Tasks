""" 
Utility functions to load and process data.

"""

import logging
import os

import torch
import numpy as np
import pandas as pd
import torch.utils.data as data

from data.datasets import TweetSentimentDataset, ImdbSentimentDataset

# Set up logging
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception:
        pass

# Load Sentiment 140 dataset
def load_sentiment_140(traindir, testdir):
    """Load the Twitter Sentiment 140 dataset"""
    train_ds = pd.read_csv(traindir)
    test_ds = pd.read_csv(testdir)
    train_ds['target'].replace({4:1},inplace=True)
    test_ds['target'].replace({4:1},inplace=True)
    X_train, y_train = train_ds.cleaned_text, train_ds.target
    X_test, y_test = test_ds.cleaned_text, test_ds.target

    return (X_train, y_train, X_test, y_test)

# Load IMDB dataset
def load_imdb_sentiment(traindir, testdir):
    train_ds = pd.read_csv(traindir)
    test_ds = pd.read_csv(testdir)
    train_ds['target'].replace({'positive':1},inplace=True)
    train_ds['target'].replace({'negative':0},inplace=True)
    test_ds['target'].replace({'positive':1},inplace=True)
    test_ds['target'].replace({'negative':0},inplace=True)

    X_train, y_train = train_ds.cleaned_text, train_ds.target
    X_test, y_test = test_ds.cleaned_text, test_ds.target

    return (X_train, y_train, X_test, y_test)

# Record the data statistics
def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list = []
    for net_id, data in net_cls_counts.items():
        n_total = 0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
        
    print("mean:", np.mean(data_list))
    print("std:", np.std(data_list))
    print("Data statistics: %s" % str(net_cls_counts))

    return net_cls_counts

# Partition the data
def partition_data(dataset, datadir, partition, n_parties, beta=0.4):
    """Data partitioning to each local party according to the beta distribution"""
    if dataset == "twitter":
        X_train, y_train, X_test, y_test = load_sentiment_140(datadir+f"{dataset}_train.csv", datadir+f"{dataset}_test.csv")
    elif dataset == "imdb":
        X_train, y_train, X_test, y_test = load_imdb_sentiment(datadir+f"{dataset}_train.csv", datadir+f"{dataset}_test.csv")

    n_train = y_train.shape[0]

    # Paritioning option
    if partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "noniid":
        min_size = 0
        min_require_size = 10
        K = 2

        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)


def get_accuracy(y_pred,y_actual):
    """Calculates the accuracy (0 to 1)

    Args:
    + y_pred (tensor ): output from the model
    + y_actual (tensor): ground truth 

    Returns:
    + float: a value between 0 to 1
    """
    y_pred = torch.argmax(y_pred, axis=1)
    return (1/len(y_actual))*torch.sum(torch.round(y_pred)==y_actual)

# Get the dataloader
def get_dataloader(datadir, batch_size, args, dataidxs=None):
    if args.dataset == "twitter":
        dl_obj = TweetSentimentDataset
    elif args.dataset == "imdb":
        dl_obj = ImdbSentimentDataset
    else:
        raise("Invalid Dataset")

    train_ds = dl_obj(
        datadir+f"{args.dataset}_train.csv",
        dataidxs=dataidxs,
        vocab_file=args.vocab,
        create_vocab=False
    )

    test_ds = dl_obj(
        datadir+f"{args.dataset}_test.csv",
        vocab_file=args.vocab,
        create_vocab=False
    )

    train_dl = data.DataLoader(dataset=train_ds, batch_size=batch_size, drop_last=True, shuffle=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False)

    return (
        train_dl,
        test_dl,
        train_ds,
        test_ds,
    )