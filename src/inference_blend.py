
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


import glob, os
import numpy as np 
import pandas as pd 
from tqdm import tqdm
from model_classes import *
from utils import *
import matplotlib.pyplot as plt

"""
was gettting really sus preds because 
the filenames from os.listdir were jumbled here 
"""
num_sub_filenames = len(os.listdir("../preds"))

"""
fixed sus preds
"""
submission_filenames = ["sub_" + str(i) + ".csv" for i in range(num_sub_filenames)]
sub_path = "../preds/"
sample_submission_columns = pd.read_csv("../input/lish-moa/sample_submission.csv").columns[1:]
sig_ids = pd.read_csv("../input/lish-moa/sample_submission.csv")["sig_id"].values


weights_df = pd.read_csv("../models/blend_weights.csv")
weights = weights_df.values.flatten()


if __name__ == "__main__":

    print("loading ",len(submission_filenames), " sub files...")

    all_preds = []

    for name in submission_filenames:
        pred = load_submission_np_array(path = sub_path + name)
        all_preds.append(pred)

    all_preds_np = np.array(all_preds)

    blend_boi = blend(all_preds_np= all_preds_np)
    final_preds = blend_boi.predict(weights= weights)


    pred_df = pd.DataFrame(final_preds, columns= sample_submission_columns)
    pred_df["sig_id"] = sig_ids
    columns_arrangement = ["sig_id"]
    columns_arrangement.extend(pred_df.columns[:-1])
    pred_df = pred_df[columns_arrangement]
    save_name = "../submission/submission.csv"
    print("saved: ", save_name)
    pred_df.to_csv(save_name, index = False)

    # plt.plot(final_preds[0])
    # plt.show()