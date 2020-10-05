
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


import glob, os
import optuna
import numpy as np 
import pandas as pd 
from tqdm import tqdm
from model_classes import *
from utils import *


filenames = [
    "model_0.pth",
    "model_1.pth",
    "model_2.pth",
    "model_3.pth",
    "model_4.pth",
    "model_5.pth",
    "model_6.pth",
    "model_100.pth",
    "model_101.pth",
    "model_102.pth",
    "model_103.pth",
    "model_104.pth",
    "model_105.pth",
    "model_106.pth",
]

model_path = "../models/"

if __name__ == "__main__":

    all_preds = []

    for i in range(len(filenames)):
        if i < 7:
            model_class = Model
            select_features = True
        else:
            model_class = Model_2
            select_features = False

        inference_model = pytorch_model(model_class = model_class, model_path = model_path + filenames[i], device = "cpu")
        x_train, y_train, x_val, y_val = load_fold(fold = "hold", train_features_path = "../input/lish-moa/train_features.csv", targets_folds_df_path = "../folds/train_targets_folds.csv", select_features= select_features)
        
        hold_set = x_val
        pred = inference_fn(inference_model, test_features = hold_set, device =  "cpu")
        loss = log_loss_metric(y_true = y_val, y_pred = pred)
        print("model ", i," log loss: ",  loss)