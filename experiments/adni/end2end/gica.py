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

def GICA_ADNI_ALL(loader, n_components=30, score_amount=3, mode="mocaz", file_mode="train"):
    Subjects = []
    Scores = []
    subjectIDs = []
    # data
    for idx, (data,info) in enumerate(loader):
        data, info = adni_collate(data, info, file_mode=file_mode)
        for m in range(data.shape[0]):
            data_m = data[m,:,:].to(device="cpu").numpy()
            Subjects.append(data_m)
            scores = mapScores(info, num=score_amount, mode=mode)
            Scores.append(scores[m])
            subjectIDs.append(info['subjectID'][m])
    # Convert to numpy arrays
    Subjects = np.array(Subjects)
    Scores = np.array(Scores).squeeze()
    num_sub, num_time, num_spatial = Subjects.shape
    #print("Num subs:", num_sub, "Num time:", num_time, "Num spatial:", num_spatial)
    Subjects = np.reshape(Subjects, (num_sub*num_time, num_spatial))
    # ---- 4. Run ICA ----
    ica = FastICA(n_components=n_components, random_state=42)
    ica_components = ica.fit_transform(Subjects) 
    #spatial_maps = ica.mixing_.T
    ica_features = ica_components
    # ---- 5. Back-Reconstruction: Split ICA component timecourses per subject ----
    subjectFeatures = []
    start = 0
    for subj in range(num_sub):
        end = start + num_time
        subj_tc = ica_features[start:end, :]  # shape: (n_timepoints, n_components)
        subjectFeatures.append(subj_tc)
        start = end
    subjectFeatures = np.array(subjectFeatures)
    # ---- 6. Extract Features for Each Subject ----
    # Mean absolute value of each component's timecourse per subject
    subjectFeatures = np.mean(np.abs(subjectFeatures), axis=1)
    #subjectFeatures = np.reshape(subjectFeatures, (num_sub, num_time * n_components))
    subjectIDs = np.array(subjectIDs).squeeze()

    return subjectFeatures, np.atleast_1d(Scores), subjectIDs

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
    C = 1.0
    gamma = 0.1
    n_components = 30
    # init folds
    files = sorted(glob.glob('/home/javier/adni/subjects/*'))
    # evaluate
    dataset = ADNI_LOADER(files, transforms=None, database=df, file_mode="all")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=workers)

    X, y, subjectIDs = GICA_ADNI_ALL(loader, n_components=n_components, score_amount=score_amount, mode=modescore, file_mode="all")
    # #np.savez_compressed(os.path.join(experiment_dir, f'iica.npz'), X=X, y=y, subjectIDs=subjectIDs)
    # data = np.load(os.path.join(experiment_dir, f'iica.npz'))
    # X = data['X']
    # y = data['y']
    # subjectIDs = data['subjectIDs']

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