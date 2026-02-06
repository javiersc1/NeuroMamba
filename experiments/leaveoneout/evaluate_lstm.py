from pathlib import Path
import numpy as np
import torch
import os
import pandas as pd
from NeuroMamba.utils.dataloaders import get_files, RSFMRI_DATALOADER, get_dataframe_entry
from NeuroMamba.utils.fmri import madc_import, score_import, mapScores, mapClasses
from torch.utils.data import Dataset, DataLoader
from scipy import stats
import time
from sklearn.model_selection import LeaveOneOut
from NeuroMamba.models.LSTM import LSTM
from sklearn.kernel_ridge import KernelRidge
from tqdm import tqdm

def evaluate(testloader, fold):
    model = LSTM(n_layers = n_layers, hidden_size= state_dim, bidirection = bidirection, dropout=dropout, score_amount=score_amount).to("cuda")
    model.load_state_dict(torch.load(os.path.join(f'/home/javier/weights/loo/lstm/{fold}_{epochs}.pth')))
    model.eval()
    realScores = 0.
    predictedScores = 0.

    for idx, (data,info) in enumerate(testloader):
        realScores = mapScores(info, num=score_amount, mode=score_mode).to("cuda", dtype=torch.float32)
        inputs = data.to("cuda", dtype=torch.float32)
        with torch.no_grad():
            predictedScores, _ = model(inputs)
    
    realScores = realScores.cpu().numpy().flatten()
    predictedScores = predictedScores.cpu().numpy().flatten()

    return realScores, predictedScores

def criterion(predictedScores, latents, scores):
    fidelity = torch.nn.functional.mse_loss(predictedScores, scores)
    reg = torch.linalg.vector_norm(latents, ord=1, dim=-1).mean()
    return fidelity + lam * reg

def leaveoneout(files):
    loo = LeaveOneOut()
    predictedScores = []
    trueScores = []
    trial = 1

    for train_index, test_index in tqdm(loo.split(files), total=loo.get_n_splits(files), desc="LOO CV"):
        train_files = files[train_index]
        test_file = files[test_index]
        # create datasets
        train_dataset = RSFMRI_DATALOADER(train_files, transforms=None, database=df, score_database=score_db)
        test_dataset = RSFMRI_DATALOADER(test_file, transforms=None, database=df, score_database=score_db)
        # create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=workers)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=workers)
        # initialize model
        torch.manual_seed(42)
        #model = LSTM(n_layers = n_layers, hidden_size= state_dim, bidirection = bidirection, dropout=dropout, score_amount=score_amount).to("cuda")
        #model.load_state_dict(torch.load(os.path.join(f'/home/javier/weights/loo/lstm/{trial}.pth')))
        # train model
        #optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.0, 0.99))
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        #scheduler = None
        # evaluate model
        real, pred = evaluate(test_loader, fold=trial)
        trueScores.append(real)
        predictedScores.append(pred)

        trial += 1

    trueScores = np.array(trueScores)
    predictedScores = np.array(predictedScores)

    return trueScores, predictedScores


if __name__ == "__main__":

    mode = "home"
    if mode == "home":
        experiment_dir = "/home/javier/Desktop/NeuroMamba/experiments/leaveoneout/"
        path = "/home/javier/madc/"
        madc_file = "/home/javier/Desktop/NeuroMamba/madc_complete.csv"
        score_file = "/home/javier/Desktop/NeuroMamba/scores.csv"
    elif mode == "server":
        experiment_dir = "/home/javiersc/NeuroMamba/experiments/leaveoneout/"
        path = "/home/javiersc/madc/"
        madc_file = "/home/javiersc/NeuroMamba/madc_complete.csv"
        score_file = "/home/javiersc/NeuroMamba/scores.csv"

    # params
    workers = 8
    df = madc_import(filename=madc_file)
    score_db = score_import(filename=score_file)
    type = "rest"
    subject_class = "remove_unknown"
    score_amount = 3
    score_mode = "mocaz"
    # model params
    n_layers = 6
    state_dim = 272
    lr = 1e-4
    lam = 0.01
    dropout = 0.0
    bidirection = True
    epochs = 5
    batch_size = 32
    batch_factor = batch_size / 1
    lr = lr * np.sqrt(batch_factor)
    # init folds
    files = np.array(get_files(path, score_db, type=type, subject_class=subject_class))
    # LOO evaluation
    real, pred = leaveoneout(files)
    # pearson correlation
    pearson = []
    pval = []
    for i in range(score_amount):
        pearson_corr = stats.pearsonr(real[:,i], pred[:,i])
        pval.append(pearson_corr.pvalue)
        pearson.append(pearson_corr.statistic)
    pearson = np.array(pearson)
    pval = np.array(pval)
    pvals = [f"{pv:.4f}" for pv in pval]
    print("Categories: [MoCA, Memory, Language]")
    print("Final Pearson: ", pearson)
    print("P-values: ", pvals)
