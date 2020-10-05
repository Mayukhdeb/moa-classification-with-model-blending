
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

sample_submission_columns = pd.read_csv("../input/lish-moa/sample_submission.csv").columns[1:]

if __name__ == "__main__":


    print("creating preds for each model on test set...")

    for i in range(len(filenames)):

        if i < 7:
            model_class = Model
            select_features = True
        else:
            model_class = Model_2
            select_features = False

        inference_model = pytorch_model(model_class = model_class, model_path = model_path + filenames[i], device = "cpu")

        test_df = preprocess(pd.read_csv("../input/lish-moa/test_features.csv"), select_features = select_features)
        sig_ids = test_df["sig_id"].values
        del test_df["sig_id"]      

        pred = inference_fn(inference_model, test_features =test_df.values, device =  "cpu")
        
        pred_df = pd.DataFrame(pred, columns= sample_submission_columns)
        pred_df["sig_id"] = sig_ids
        
        columns_arrangement = ["sig_id"]
        columns_arrangement.extend(pred_df.columns[:-1])
        pred_df = pred_df[columns_arrangement]
        save_name = "../preds/sub_" + str(i) + ".csv"
        print("saved: ", save_name)
        pred_df.to_csv(save_name, index = False)

        