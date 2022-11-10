import torch
import torch.nn as nn

# AutoGluon and HPO tools
import autogluon.core as ag
import pandas as pd
import numpy as np
import random
import math
from embedder import NLP_embedder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
import time
from data import load_data, SimpleDataset, load_wiki
from torch.utils.data import DataLoader
from plot import plot_TSNE_clustering
# Fixing seed for reproducibility
SEED = 999
random.seed(SEED)
np.random.seed(SEED)

ACTIVE_METRIC_NAME = 'accuracy'
REWARD_ATTR_NAME = 'objective'
datasets = [ "mnli","cola", "sst2", "mrpc","qqp", "rte"]#"qqp", "rte" 
eval_ds = [ "rtesmall", "qqpsmall","qqp", "rte"]
    
 
            

def train(args, config):
    max_epochs = int(config["DEFAULT"]["epochs"])

    batch_size = int(config["DEFAULT"]["batch_size"])

    dataset = config["DEFAULT"]["dataset"]
    optimizer = config["DEFAULT"]["optim"]


    print("dataset:", dataset)
    lr = 2e-5

    
    print("running baseline")
    args.number_of_diff_lrs = int(config["DEFAULT"]["num_diff_opt"])
    args.opts = {"lr": lr, "opt": optimizer}
    args.ds = dataset
    args.split_by = config["DEFAULT"]["split_by"]
    args.model = config["DEFAULT"]["model"]
    args.savepth = config["DEFAULT"]["directory"]
    num_classes = 2
    if "mnli" in dataset:
        num_classes = 3

    print("loading model")
    model = NLP_embedder(num_classes = num_classes,batch_size = batch_size,args =  args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print("loading dataset")
    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_data(name=dataset)
    print("training model on dataset", dataset)
    model.fit(X_train, Y_train, epochs=max_epochs, X_val= X_val, Y_val = Y_val)
    accuracy = model.evaluate(X_val,Y_val).item()
    print("acuraccy on ds:", accuracy)
       