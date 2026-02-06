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
from NeuroMamba.models.FCM import FCM
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from NeuroMamba.models.GICA import GICA
from sklearn.svm import SVC

def get_scores_and_labels(loader):
    scores_list = []
    labels_list = []
    for _, info in loader:
        scores = info['cognitionz'].numpy()
        labels = info['label']
        scores_list.append(scores)
        labels_list.append(labels)
    scores_array = np.vstack(scores_list)
    labels_array = np.hstack(labels_list)
    return scores_array, labels_array

def model_scoreonly(scores, labels):
    loo = LeaveOneOut()
    predictedLabels = []
    trueLabels = mapClasses(labels, 2)
    trial = 1

    for train_index, test_index in tqdm(loo.split(scores), total=loo.get_n_splits(scores), desc="LOO"):
        trainScore, testScore = scores[train_index,:], scores[test_index,:]
        trainLabels, testLabels = trueLabels[train_index], trueLabels[test_index]
        clf = LogisticRegression(random_state=42, C=1, class_weight='balanced').fit(trainScore, trainLabels)
        #clf = SVC(probability=True, random_state=42, C=1.0, kernel='rbf', gamma='auto').fit(trainScore, trainLabels)
        predictions = clf.predict_proba(testScore)
        predictedLabels.append(predictions)
        trial += 1

    predictedLabels = np.array(predictedLabels).squeeze()

    return trueLabels, predictedLabels[:,-1]  # return probabilities for the positive class

def model_features(scores, labels, X, C = 0.1, normalize=True):
    loo = LeaveOneOut()
    predictedLabels = []
    trueLabels = mapClasses(labels, 2)
    trial = 1
    finalLabels = []

    # normalize X by z-score\
    if normalize:
        X = stats.zscore(X, axis=0)

    # concatenate scores and X
    features = np.hstack([scores, X])

    for train_index, test_index in tqdm(loo.split(features), total=loo.get_n_splits(features), desc="LOO"):
        trainFeatures, testFeatures = features[train_index,:], features[test_index,:]
        trainLabels, testLabels = trueLabels[train_index], trueLabels[test_index]
        clf = LogisticRegression(random_state=42, C=C, class_weight='balanced').fit(trainFeatures, trainLabels)
        #clf = SVC(probability=True, random_state=42, C=C, kernel='rbf', gamma='scale', shrinking=False).fit(trainFeatures, trainLabels)
        predictions = clf.predict_proba(testFeatures)
        predictedLabels.append(predictions)
        finalLabels.append(testLabels)
        trial += 1

    predictedLabels = np.array(predictedLabels).squeeze()
    finalLabels = np.array(finalLabels).squeeze()

    return finalLabels, predictedLabels[:,-1]  # return probabilities for the positive class


if __name__ == "__main__":

    mode = "home"
    if mode == "home":
        experiment_dir = "/home/javier/Desktop/DeepScore/experiments/roc/"
        figure_dir = "/home/javier/Desktop/DeepScore/figures/"
        path = "/home/javier/madc/"
        madc_file = "/home/javier/Desktop/DeepScore/madc_complete.csv"
        score_file = "/home/javier/Desktop/DeepScore/scores.csv"
    elif mode == "server":
        experiment_dir = "/home/javiersc/DeepScore/experiments/roc/"
        figure_dir = "/home/javiersc/DeepScore/figures/"
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
    # init folds
    files = np.array(get_files(path, df, type=type, subject_class=subject_class))
    # evaluate
    dataset = RSFMRI_DATALOADER(files, transforms=None, database=df, score_database=score_db)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=workers)

    # MOCA only
    scores, labels = get_scores_and_labels(loader)
    true_labels_moca, pred_probs_moca = model_scoreonly(scores, labels)
    # # save to csv file
    # results_df = pd.DataFrame({
    #     'PredictedProbability': pred_probs_moca
    # })
    # results_df.to_csv(os.path.join(experiment_dir, 'moca.csv'), index=False)
    fpr_moca, tpr_moca, _ = roc_curve(true_labels_moca, pred_probs_moca)
    roc_auc_moca = auc(fpr_moca, tpr_moca)

    # FCM
    X, _ = FCM(loader, score_amount=score_amount, mode=modescore)
    # perform PCA on X
    pca = PCA(n_components=X.shape[0], random_state=42, whiten=True)
    X = pca.fit_transform(X)
    true_labels_fcm, pred_probs_fcm = model_features(scores, labels, X, C=1.0) # 1.0 for LR
    fpr_fcm, tpr_fcm, _ = roc_curve(true_labels_fcm, pred_probs_fcm)
    roc_auc_fcm = auc(fpr_fcm, tpr_fcm)

    # G-ICA
    X, _ = GICA(loader, n_components=30, score_amount=score_amount, mode=modescore)
    true_labels_gica, pred_probs_gica = model_features(scores, labels, X, C=0.01, normalize=False) # 0.01 for LR
    fpr_gica, tpr_gica, _ = roc_curve(true_labels_gica, pred_probs_gica)
    roc_auc_gica = auc(fpr_gica, tpr_gica)

    # NeuroMamba
    mamba_df = pd.read_csv(os.path.join("/home/javier/Desktop/DeepScore/experiments/cross_entropy/", 'neuromamba_original.csv'))
    true_labels_neuromamba, pred_probs_neuromamba = mamba_df['Label'].values, mamba_df['Predicted_Probability'].values
    fpr_neuromamba, tpr_neuromamba, _ = roc_curve(true_labels_neuromamba, pred_probs_neuromamba)
    roc_auc_neuromamba = auc(fpr_neuromamba, tpr_neuromamba)

    # results
    print(f"MoCA AUC: {roc_auc_moca:.4f}")
    print(f"FCM AUC: {roc_auc_fcm:.4f}")
    print(f"G-ICA AUC: {roc_auc_gica:.4f}")
    print(f"NeuroMamba AUC: {roc_auc_neuromamba:.4f}")

    colors = sns.color_palette(palette='Set1')

    # Plot ROC curve
    sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True)
    plt.figure(figsize=(6, 6))
    sns.lineplot(x=fpr_moca, y=tpr_moca, label=f'MoCA (AUC = {roc_auc_moca:.2f})', linewidth=2, errorbar=None, color='black')
    sns.lineplot(x=fpr_fcm, y=tpr_fcm, label=f'MoCA + FCM (AUC = {roc_auc_fcm:.2f})', linewidth=2, errorbar=None, linestyle=':', alpha=0.99, color=colors[1])
    sns.lineplot(x=fpr_gica, y=tpr_gica, label=f'MoCA + GICA (AUC = {roc_auc_gica:.2f})', linewidth=2, errorbar=None, linestyle='-.', alpha=0.99, color=colors[2])
    sns.lineplot(x=fpr_neuromamba, y=tpr_neuromamba, label=f'NeuroMamba-BCE (AUC = {roc_auc_neuromamba:.2f})', linewidth=2, errorbar=None, linestyle='--', alpha=1.0, color=colors[0])
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'roc.pdf'), dpi=600, bbox_inches='tight')