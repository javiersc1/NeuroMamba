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
from NeuroMamba.utils.adni import *
from NeuroMamba.models.CPM import CPM_ADNI
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from tqdm import tqdm
from sklearn.decomposition import FastICA
from NeuroMamba.utils.fmri import generateStaticFC, matrix2vec

def model(X, y, subjectIDs, C=200, gamma=0.01):
    predictedScores = []
    trueScores = []
    unique_subjects = np.unique(subjectIDs)

    for subjectID in tqdm(unique_subjects, total=len(unique_subjects), desc="CV"):

        trainFeatures, testFeatures = X[subjectIDs != subjectID,:], X[subjectIDs == subjectID,:]
        trainScore, testScore = y[subjectIDs != subjectID], y[subjectIDs == subjectID]
        
        clf = KernelRidge(alpha=C, kernel='rbf', gamma=gamma)
        clf.fit(trainFeatures, trainScore)
        pred = clf.predict(testFeatures)
        predictedScores.append(pred)
        trueScores.append(testScore)

    predictedScores = np.concatenate(predictedScores).squeeze()
    trueScores = np.concatenate(trueScores).squeeze()

    return trueScores, predictedScores


if __name__ == "__main__":

    mode = "home"
    if mode == "home":
        experiment_dir = "/home/javier/Desktop/NeuroMamba/experiments/adni/end2end/"
        path = "/home/javier/adni/"
        madc_file = "/home/javier/Desktop/NeuroMamba/madc_complete.csv"
        score_file = "/home/javier/Desktop/NeuroMamba/scores.csv"
    elif mode == "server":
        experiment_dir = "/home/javiersc/DeepScore/experiments/adni/end2end/"
        path = "/home/javiersc/adni/"
        madc_file = "/home/javiersc/DeepScore/madc_complete.csv"
        score_file = "/home/javiersc/DeepScore/scores.csv"

    # params
    workers = 8
    df = adni_import("/home/javier/Desktop/NeuroMamba/adni.csv")
    score_amount = 1
    modescore = "mocaz"
    # model params
    C = 0.1
    gamma = 0.0001
    # init folds
    files = sorted(glob.glob('/home/javier/adni/subjects/*'))
    # evaluate
    dataset = ADNI_LOADER(files, transforms=None, database=df, file_mode="train")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=workers)

    # load files
    X = np.load('/home/javier/weights/cpm+_subjectFeatures_adni.npy')
    y = np.load('/home/javier/weights/cpm+_Scores_adni.npy')
    subjectIDs = np.arange(X.shape[0])  # Assuming each row corresponds to a unique subject, we can use indices as subject IDs

    print(f"Feature matrix shape: {X.shape}")
    print(f"Score matrix shape: {y.shape}")
    print(f"Subjects: {len(subjectIDs)}")
    print(f"Unique Subjects: {len(np.unique(subjectIDs))}")
    real, pred = model(X, y, subjectIDs, C=C, gamma=gamma)
    print(real.shape, pred.shape)
    # pearson correlation
    pearson_corr = stats.pearsonr(real, pred)
    pval = pearson_corr.pvalue
    pearson = pearson_corr.statistic
    print("Categories: [MoCA]")
    print("Final Pearson: ", pearson)
    print("P-values: ", pval)