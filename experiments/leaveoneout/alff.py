from pathlib import Path
import numpy as np
import torch
import os
import pandas as pd
from NeuroMamba.utils.dataloaders import get_files, RSFMRI_DATALOADER, get_dataframe_entry
from NeuroMamba.utils.fmri import madc_import, score_import, mapScores, mapClasses
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import SVR
from scipy import stats
import time
from sklearn.model_selection import LeaveOneOut
from NeuroMamba.models.ALFF import ALFF
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
#from sklearn import linear_model
from tqdm import tqdm

def model(X, y, C=200, gamma=0.01):
    loo = LeaveOneOut()
    predictedScores = []
    trial = 1

    for train_index, test_index in tqdm(loo.split(X), total=loo.get_n_splits(X), desc="LOO CV"):
        #print(f"Trial: {trial}")
        trainICA, testICA = X[train_index], X[test_index]
        trainScore, testScore = y[train_index], y[test_index]
        predictions = []
        for i in range(score_amount):
            clf = KernelRidge(alpha=C, kernel='linear', gamma=gamma)
            #clf = SVR(C=C, gamma='auto')
            clf.fit(trainICA, trainScore[:,i])
            pred = clf.predict(testICA)
            predictions.append(pred)

        predictions = np.array(predictions).flatten()
        predictedScores.append(predictions)
        trial += 1

    predictedScores = np.array(predictedScores)
    trueScores = y

    return trueScores, predictedScores


if __name__ == "__main__":

    mode = "home"
    if mode == "home":
        experiment_dir = "/home/javier/Desktop/NeuroMamba/experiments/leaveoneout/"
        path = "/home/javier/madc/"
        madc_file = "/home/javier/Desktop/NeuroMamba/madc_complete.csv"
        score_file = "/home/javier/Desktop/NeuroMamba/scores.csv"
    elif mode == "server":
        experiment_dir = "/home/javiersc/DeepScore/experiments/leaveoneout/"
        path = "/home/javiersc/madc/"
        madc_file = "/home/javiersc/DeepScore/madc_complete.csv"
        score_file = "/home/javiersc/DeepScore/scores.csv"

    # params
    workers = 8
    df = madc_import(filename=madc_file)
    score_db = score_import(filename=score_file)
    type = "rest"
    subject_class = "remove_unknown"
    score_amount = 3
    modescore = "mocaz"
    # model params
    C = 100
    gamma = 0.1
    # init folds
    files = np.array(get_files(path, score_db, type=type, subject_class=subject_class))
    predictions = []
    true = []
    trial = 0
    # evaluate

    dataset = RSFMRI_DATALOADER(files, transforms=None, database=df, score_database=score_db)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=False, num_workers=workers)
    X, y = ALFF(loader, score_amount=score_amount, mode=modescore)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Score matrix shape: {y.shape}")
    real, pred = model(X, y, C=C, gamma=gamma)

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
    print(f"Final Pearson: {pearson}")
    print(f"P-values: {pvals}")