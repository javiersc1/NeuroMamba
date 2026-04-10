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
from NeuroMamba.models.FCM import FCM_ADNI
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
#from sklearn import linear_model
from tqdm import tqdm
from sklearn.decomposition import FastICA
from NeuroMamba.utils.fmri import generateStaticFC, matrix2vec

def IICA_ADNI_ALL(loader, n_components=30, score_amount=3, mode="mocaz", file_mode="train"):
    SubjectFeatures = []
    Scores = []
    subjectIDs = []
    trial = 1

    for idx, (data,info) in enumerate(loader):
        print(f"Processing subject {trial}")
        data, info = adni_collate(data, info, file_mode=file_mode)
        for m in range(data.shape[0]):
            data_m = data[m,:,:].to(device="cpu").numpy()
            #print("Data shape: ", data_m.shape)
            transformer = FastICA(n_components=n_components,random_state=42,max_iter=1000, algorithm='deflation')
            transformer.fit(data_m.T)
            components = transformer.components_
            features = np.mean(np.abs(components), axis=1)
            SubjectFeatures.append(features)
            scores = mapScores(info, num=score_amount, mode=mode)
            Scores.append(scores[m])
            subjectIDs.append(info['subjectID'][m])
        trial += 1

    Scores = np.array(Scores).squeeze()
    SubjectFeatures = np.stack(SubjectFeatures)
    subjectIDs = np.array(subjectIDs).squeeze()
    
    return SubjectFeatures, np.atleast_1d(Scores), subjectIDs

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
        experiment_dir = "/home/javier/Desktop/DeepScore/experiments/adni/end2end/"
        path = "/home/javier/adni/"
        madc_file = "/home/javier/Desktop/DeepScore/madc_complete.csv"
        score_file = "/home/javier/Desktop/DeepScore/scores.csv"
    elif mode == "server":
        experiment_dir = "/home/javiersc/DeepScore/experiments/adni/end2end/"
        path = "/home/javiersc/adni/"
        madc_file = "/home/javiersc/DeepScore/madc_complete.csv"
        score_file = "/home/javiersc/DeepScore/scores.csv"

    # params
    workers = 8
    df = adni_import()
    score_amount = 1
    modescore = "mocaz"
    # model params
    C = 0.1
    gamma = 10
    n_components = 30
    # init folds
    files = sorted(glob.glob('/home/javier/adni/subjects/*'))
    # evaluate
    dataset = ADNI_LOADER(files, transforms=None, database=df, file_mode="all")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=workers)

    #X, y, subjectIDs = IICA_ADNI_ALL(loader, n_components=n_components, score_amount=score_amount, mode=modescore, file_mode="all")
    #np.savez_compressed(os.path.join(experiment_dir, f'iica.npz'), X=X, y=y, subjectIDs=subjectIDs)
    data = np.load(os.path.join(experiment_dir, f'iica.npz'))
    X = data['X']
    y = data['y']
    subjectIDs = data['subjectIDs']

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