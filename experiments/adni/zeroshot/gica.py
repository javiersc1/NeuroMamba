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
#from DeepScore.models.GICA import GICA_ADNI
from NeuroMamba.models.GICA import GICA
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
#from sklearn import linear_model
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA

def model(X_train, y_train, X_test, y_test, C=200, gamma=0.01):
    clf = KernelRidge(alpha=C, kernel='rbf', gamma=gamma)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    true = y_test
    return true, pred


def GICA_COMBINED(adni_loader, madc_loader, n_components=30, score_amount=3, mode="mocaz", file_mode="train"):
    adniSubjects = []
    adniScores = []
    # ADNI data
    for idx, (data,info) in enumerate(adni_loader):
        data, info = adni_collate(data, info, file_mode=file_mode)
        for m in range(data.shape[0]):
            data_m = data[m,:,:].to(device="cpu").numpy()
            adniSubjects.append(data_m)
            scores = mapScores(info, num=score_amount, mode=mode)
            adniScores.append(scores[m])
    # Convert to numpy arrays
    adniSubjects = np.array(adniSubjects)
    adniScores = np.array(adniScores).squeeze()
    B_adni, T_adni, R_adni = adniSubjects.shape

    # MADC data
    madcSubjects = []
    madcScores = []
    # data
    for idx, (data,info) in enumerate(madc_loader):
        data = data[0,:,:].to(device="cpu").numpy()
        madcSubjects.append(data)
        scores = mapScores(info, num=score_amount, mode=mode)
        madcScores.append(scores)
    # Convert to numpy arrays
    madcSubjects = np.array(madcSubjects)
    madcScores = np.array(madcScores).squeeze()
    B_madc, T_madc, R_madc = madcSubjects.shape

    # Temporal concat
    adniSubjects = np.reshape(adniSubjects, (B_adni*T_adni, R_adni))
    madcSubjects = np.reshape(madcSubjects, (B_madc*T_madc, R_madc))
    Subjects = np.concatenate((adniSubjects, madcSubjects), axis=0)
    
    # ---- 4. Run ICA ----
    ica = FastICA(n_components=n_components, random_state=42)
    ica_components = ica.fit_transform(Subjects) 

    # unconcatenate ADNI and MADC
    adni_ica_features = ica_components[0:(B_adni*T_adni), :]
    madc_ica_features = ica_components[(B_adni*T_adni):, :]

    # ---- 5. Back-Reconstruction: Split ICA component timecourses per subject ----
    adniFeatures = []
    start = 0
    for subj in range(B_adni):
        end = start + T_adni
        subj_tc = adni_ica_features[start:end, :]  # shape: (n_timepoints, n_components)
        adniFeatures.append(subj_tc)
        start = end
    adniFeatures = np.array(adniFeatures)

    madcFeatures = []
    start = 0
    for subj in range(B_madc):
        end = start + T_madc
        subj_tc = madc_ica_features[start:end, :]  # shape: (n_timepoints, n_components)
        madcFeatures.append(subj_tc)
        start = end
    madcFeatures = np.array(madcFeatures)

    # ---- 6. Extract Features for Each Subject ----
    # Mean absolute value of each component's timecourse per subject
    adniFeatures = np.mean(np.abs(adniFeatures), axis=1)
    madcFeatures = np.mean(np.abs(madcFeatures), axis=1)
    #subjectFeatures = np.reshape(subjectFeatures, (num_sub, num_time * n_components))

    return madcFeatures,  np.atleast_1d(madcScores), adniFeatures, np.atleast_1d(adniScores)


if __name__ == "__main__":

    mode = "home"
    if mode == "home":
        experiment_dir = "/home/javier/Desktop/DeepScore/experiments/adni/zeroshot/"
        path = "/home/javier/adni/"
        madc_path = "/home/javier/madc/"
        madc_file = "/home/javier/Desktop/DeepScore/madc_complete.csv"
        score_file = "/home/javier/Desktop/DeepScore/scores.csv"
    elif mode == "server":
        experiment_dir = "/home/javiersc/DeepScore/experiments/adni/zeroshot/"
        path = "/home/javiersc/adni/"
        madc_file = "/home/javiersc/DeepScore/madc_complete.csv"
        score_file = "/home/javiersc/DeepScore/scores.csv"

    # params
    workers = 8
    adni_df = adni_import()
    score_amount = 1
    modescore = "mocaz"
    # model params
    C = 1.0
    gamma = 0.1
    n_components = 30
    # MADC DATA
    madc_df = madc_import(filename=madc_file)
    madc_score = score_import(filename=score_file)
    madc_files = np.array(get_files(madc_path, madc_df, type="rest", subject_class="remove_unknown"))
    madc_dataset = RSFMRI_DATALOADER(madc_files, transforms=None, database=madc_df, score_database=madc_score)
    madc_loader = DataLoader(madc_dataset, batch_size=1, shuffle=True, drop_last=False, num_workers=workers)
    #X_train, y_train = GICA(madc_loader, n_components=n_components, score_amount=score_amount, mode=modescore)
    # adni evaluate
    adni_files = sorted(glob.glob('/home/javier/adni/subjects/*'))
    adni_dataset = ADNI_LOADER(adni_files, transforms=None, database=adni_df, file_mode="all")
    adni_loader = DataLoader(adni_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=workers)
    X_train, y_train, X_test, y_test = GICA_COMBINED(adni_loader, madc_loader, n_components=n_components, score_amount=score_amount, mode=modescore, file_mode="all")
    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Score matrix shape: {y_train.shape}")
    print(f"Feature matrix shape: {X_test.shape}")
    print(f"Score matrix shape: {y_test.shape}")
    # zero shot transfer
    real, pred = model(X_train, y_train, X_test, y_test, C=C, gamma=gamma)
    print(real.shape, pred.shape)
    # pearson correlation
    pearson_corr = stats.pearsonr(real, pred)
    pval = pearson_corr.pvalue
    pearson = pearson_corr.statistic
    print("Categories: [MoCA]")
    print("Final Pearson: ", pearson)
    print("P-values: ", pval)