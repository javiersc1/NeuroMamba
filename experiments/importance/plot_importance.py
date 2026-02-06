from pathlib import Path
import numpy as np
import torch
import os
import pandas as pd
# from DeepScore.utils.dataloaders import get_dataloader
from NeuroMamba.utils.fmri import madc_import, power_import, score_import, mapScores
from scipy import stats
import time
import matplotlib.pyplot as plt
import seaborn as sns
#from DeepScore.utils.trainpost import extract_features
from NeuroMamba.models.NeuroMamba import NeuroMamba
from sklearn.model_selection import LeaveOneOut
from NeuroMamba.utils.dataloaders import get_files, RSFMRI_DATALOADER
from torch.utils.data import Dataset, DataLoader
from sklearn import linear_model
from sklearn.inspection import permutation_importance
from tqdm import tqdm

mode = "home"
if mode == "home":
    project_dir = "/home/javier/Desktop/DeepScore/"
    madc_file = "/home/javier/Desktop/DeepScore/madc_complete.csv"
    power_file = "/home/javier/Desktop/DeepScore/power_atlas.csv"
    path="/home/javier/madc/"
    score_file = "/home/javier/Desktop/DeepScore/scores.csv"
else:
    project_dir = "/home/javiersc/DeepScore/"
    madc_file = "/home/javiersc/DeepScore/madc_complete.csv"
    power_file = "/home/javiersc/DeepScore/power_atlas.csv"
    path="/home/javiersc/madc/"
    score_file = "/home/javiersc/DeepScore/scores.csv"

power = power_import(power_file)
df = madc_import(madc_file)
coords = power[['X', 'Y', 'Z']].values
score_db = score_import(filename=score_file)

files = np.array(get_files(path, df, type="rest", subject_class="remove_unknown"))
loo = LeaveOneOut()
fold = 1

for train_index, test_index in tqdm(loo.split(files), total=loo.get_n_splits(files), desc="EVALUATE"):
    train_files = files[train_index]
    test_file = files[test_index]
    # create datasets
    train_dataset = RSFMRI_DATALOADER(train_files, transforms=None, database=df, score_database=score_db)
    test_dataset = RSFMRI_DATALOADER(test_file, transforms=None, database=df, score_database=score_db)
    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=8)
    # evaluate model
    model = NeuroMamba(n_layers=12, state_dim=32, num_variables=272, score_amount=3).to("cuda")
    model.load_state_dict(torch.load(os.path.join(f'/home/javier/Desktop/DeepScore/weights/loo/neuromamba/{fold}_100.pth')))
    model.eval()
    # data extraction
    trueScoreTrain = []
    predScoreTrain = []
    trueScoreTest = []
    predScoreTest = []
    latentsTest = []
    latentsTrain = []
    importances_moca = []
    importances_memory = []
    importances_language = []

    for idx, (data,info) in enumerate(train_loader):
            scores = mapScores(info, num=3, mode="mocaz").to("cuda", dtype=torch.float32)
            inputs = data.to("cuda", dtype=torch.float32)
            with torch.no_grad():
                predictedScores, latents, _ = model.evaluate(inputs)
            predictedScores = predictedScores.cpu().numpy()
            trueScores = scores.cpu().numpy()
            latents = latents.cpu().numpy()
            trueScoreTrain.append(trueScores)
            predScoreTrain.append(predictedScores)
            latentsTrain.append(latents)

    for idx, (data,info) in enumerate(test_loader):
            scores = mapScores(info, num=3, mode="mocaz").to("cuda", dtype=torch.float32)
            inputs = data.to("cuda", dtype=torch.float32)
            with torch.no_grad():
                predictedScores, latents, _ = model.evaluate(inputs)
            predictedScores = predictedScores.cpu().numpy()
            trueScores = scores.cpu().numpy()
            latents = latents.cpu().numpy()
            trueScoreTest.append(trueScores)
            predScoreTest.append(predictedScores)
            latentsTest.append(latents)

    latentsTrain = np.array(latentsTrain).squeeze()
    latentsTest = np.array(latentsTest).squeeze()
    trueScoreTrain = np.vstack(trueScoreTrain)
    predScoreTrain =  np.vstack(predScoreTrain)
    trueScoreTest =  np.vstack(trueScoreTest)
    predScoreTest =  np.vstack(predScoreTest)

    clf = linear_model.Lasso(alpha=0.01,max_iter=10000)
    clf.fit(latentsTrain, trueScoreTrain[:,0])
    r = permutation_importance(clf, latentsTrain, trueScoreTrain[:,0], n_repeats=100, random_state=42, scoring='neg_root_mean_squared_error')
    importance = r.importances_mean
    importances_moca.append(importance)

    clf = linear_model.Lasso(alpha=0.01,max_iter=10000)
    clf.fit(latentsTrain, trueScoreTrain[:,1])
    r = permutation_importance(clf, latentsTrain, trueScoreTrain[:,1], n_repeats=100, random_state=42, scoring='neg_root_mean_squared_error')
    importance = r.importances_mean
    importances_memory.append(importance)

    clf = linear_model.Lasso(alpha=0.01,max_iter=10000)
    clf.fit(latentsTrain, trueScoreTrain[:,2])
    r = permutation_importance(clf, latentsTrain, trueScoreTrain[:,2], n_repeats=100, random_state=42, scoring='neg_root_mean_squared_error')
    importance = r.importances_mean
    importances_language.append(importance)
    
    # next fold
    fold += 1

    # if fold == 5:
    #     break

# collect importances
importances_moca = np.array(importances_moca)
mean_importance_moca = np.mean(importances_moca, axis=0)
importances_memory = np.array(importances_memory)
mean_importance_memory = np.mean(importances_memory, axis=0)
importances_language = np.array(importances_language)
mean_importance_language = np.mean(importances_language, axis=0)

# MoCA Report

filtered_df = power
filtered_df["FPI"] = mean_importance_moca
filtered_df = filtered_df.nlargest(5, 'FPI')
df_sorted = filtered_df.sort_values(by='FPI', key=lambda x: np.abs(x), ascending=False)
df_sorted = df_sorted.drop(['Color'], axis=1)
df_sorted = df_sorted.drop(['Master Assignment'], axis=1)
df_sorted = df_sorted.drop(['Matter'], axis=1)
df_sorted = df_sorted.drop(['TD Brodman Area'], axis=1)
df_sorted = df_sorted.drop(['Brodmann area'], axis=1)
df_sorted = df_sorted.drop(['Cerebrum'], axis=1)
df_sorted.rename(columns={'Power Suggested System': 'Nominal System'}, inplace=True)
df_sorted.rename(columns={'Lobe': 'Lobe/Area'}, inplace=True)
df_sorted.rename(columns={'TD Label': 'Talairach Daemon Label'}, inplace=True)
df_sorted['MNI (x,y,z)'] = list(zip(df_sorted['X'], df_sorted['Y'], df_sorted['Z']))
df_sorted = df_sorted.drop(['X'], axis=1)
df_sorted = df_sorted.drop(['Y'], axis=1)
df_sorted = df_sorted.drop(['Z'], axis=1)
column = df_sorted.pop('MNI (x,y,z)')
df_sorted.insert(1, 'MNI (x,y,z)', column)
column = df_sorted.pop('FPI')
df_sorted.insert(0, 'FPI', column)
df_sorted.rename(columns={'FPI': 'FPI (RMSE)'}, inplace=True)
df_sorted.rename(columns={'ROI': 'Power ROI'}, inplace=True)
df_sorted.to_latex(project_dir+'figures/importance_moca.tex', index=False, float_format="%.2f")
df_sorted

# Memory Report

filtered_df = power
filtered_df["FPI"] = mean_importance_memory
filtered_df = filtered_df.nlargest(5, 'FPI')
df_sorted = filtered_df.sort_values(by='FPI', key=lambda x: np.abs(x), ascending=False)
df_sorted = df_sorted.drop(['Color'], axis=1)
df_sorted = df_sorted.drop(['Master Assignment'], axis=1)
df_sorted = df_sorted.drop(['Matter'], axis=1)
df_sorted = df_sorted.drop(['TD Brodman Area'], axis=1)
df_sorted = df_sorted.drop(['Brodmann area'], axis=1)
df_sorted = df_sorted.drop(['Cerebrum'], axis=1)
df_sorted.rename(columns={'Power Suggested System': 'Nominal System'}, inplace=True)
df_sorted.rename(columns={'Lobe': 'Lobe/Area'}, inplace=True)
df_sorted.rename(columns={'TD Label': 'Talairach Daemon Label'}, inplace=True)
df_sorted['MNI (x,y,z)'] = list(zip(df_sorted['X'], df_sorted['Y'], df_sorted['Z']))
df_sorted = df_sorted.drop(['X'], axis=1)
df_sorted = df_sorted.drop(['Y'], axis=1)
df_sorted = df_sorted.drop(['Z'], axis=1)
column = df_sorted.pop('MNI (x,y,z)')
df_sorted.insert(1, 'MNI (x,y,z)', column)
column = df_sorted.pop('FPI')
df_sorted.insert(0, 'FPI', column)
df_sorted.rename(columns={'FPI': 'FPI (RMSE)'}, inplace=True)
df_sorted.rename(columns={'ROI': 'Power ROI'}, inplace=True)
df_sorted.to_latex(project_dir+'figures/importance_memory.tex', index=False, float_format="%.2f")
df_sorted

# Language Report

filtered_df = power
filtered_df["FPI"] = mean_importance_language
filtered_df = filtered_df.nlargest(5, 'FPI')
df_sorted = filtered_df.sort_values(by='FPI', key=lambda x: np.abs(x), ascending=False)
df_sorted = df_sorted.drop(['Color'], axis=1)
df_sorted = df_sorted.drop(['Master Assignment'], axis=1)
df_sorted = df_sorted.drop(['Matter'], axis=1)
df_sorted = df_sorted.drop(['TD Brodman Area'], axis=1)
df_sorted = df_sorted.drop(['Brodmann area'], axis=1)
df_sorted = df_sorted.drop(['Cerebrum'], axis=1)
df_sorted.rename(columns={'Power Suggested System': 'Nominal System'}, inplace=True)
df_sorted.rename(columns={'Lobe': 'Lobe/Area'}, inplace=True)
df_sorted.rename(columns={'TD Label': 'Talairach Daemon Label'}, inplace=True)
df_sorted['MNI (x,y,z)'] = list(zip(df_sorted['X'], df_sorted['Y'], df_sorted['Z']))
df_sorted = df_sorted.drop(['X'], axis=1)
df_sorted = df_sorted.drop(['Y'], axis=1)
df_sorted = df_sorted.drop(['Z'], axis=1)
column = df_sorted.pop('MNI (x,y,z)')
df_sorted.insert(1, 'MNI (x,y,z)', column)
column = df_sorted.pop('FPI')
df_sorted.insert(0, 'FPI', column)
df_sorted.rename(columns={'FPI': 'FPI (RMSE)'}, inplace=True)
df_sorted.rename(columns={'ROI': 'Power ROI'}, inplace=True)
df_sorted.to_latex(project_dir+'figures/importance_language.tex', index=False, float_format="%.2f")
df_sorted