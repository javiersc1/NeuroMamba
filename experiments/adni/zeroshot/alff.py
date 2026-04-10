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
from NeuroMamba.models.ALFF import ALFF_ADNI
from NeuroMamba.models.ALFF import ALFF
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
#from sklearn import linear_model
from tqdm import tqdm
from sklearn.decomposition import PCA

def model(X_train, y_train, X_test, y_test, C=200, gamma=0.01):
    clf = KernelRidge(alpha=C, kernel='linear', gamma=gamma)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    true = y_test
    return true, pred


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
    C = 100
    gamma = 0.1
    # MADC DATA
    madc_df = madc_import(filename=madc_file)
    madc_score = score_import(filename=score_file)
    madc_files = np.array(get_files(madc_path, madc_df, type="rest", subject_class="remove_unknown"))
    madc_dataset = RSFMRI_DATALOADER(madc_files, transforms=None, database=madc_df, score_database=madc_score)
    madc_loader = DataLoader(madc_dataset, batch_size=1, shuffle=True, drop_last=False, num_workers=workers)
    X_train, y_train = ALFF(madc_loader, score_amount=score_amount, mode=modescore)
    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Score matrix shape: {y_train.shape}")
    # adni evaluate
    adni_files = sorted(glob.glob('/home/javier/adni/subjects/*'))
    adni_dataset = ADNI_LOADER(adni_files, transforms=None, database=adni_df, file_mode="all")
    adni_loader = DataLoader(adni_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=workers)
    X_test, y_test = ALFF_ADNI(adni_loader, score_amount=score_amount, mode=modescore, file_mode="all")
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