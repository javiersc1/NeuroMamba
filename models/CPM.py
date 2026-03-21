from tqdm import tqdm

import torch
import torch.nn as nn
import math
from torchinfo import summary
from NeuroMamba.utils.fmri import *
from NeuroMamba.utils.dataloaders import get_files, RSFMRI_DATALOADER, get_dataframe_entry
from NeuroMamba.utils.fmri import madc_import, score_import, mapScores, mapClasses
from torch.utils.data import Dataset, DataLoader
from NeuroMamba.utils.adni import adni_collate
from scipy.stats import pearsonr

def CPM(loader, score_amount=3, mode="mocaz", edge_type="positive", thresh=0.01):
    FC = []
    Scores = []
    masks = []
    subjectFeatures = []

    for idx, (data,info) in enumerate(loader):
        x = data.to(device="cpu")
        x = generateStaticFC(x)
        x = matrix2vec(x)
        FC.append(x.numpy())
        score = mapScores(info, num=score_amount, mode=mode)
        Scores.append(score)

    Scores = np.array(Scores).squeeze()
    FC = np.array(FC).squeeze()

    # CPM
    for i in range(score_amount):
        y = Scores[:, i]
        features, mask = cpm_method(FC, y, thresh=thresh, edge_type=edge_type)
        subjectFeatures.append(features)
        masks.append(mask)

    subjectFeatures = np.concatenate(subjectFeatures, axis=1)
    masks = np.stack(masks, axis=1)

    return subjectFeatures, Scores, masks

def cpm_feature_selection(X, y, thresh=0.01, edge_type="positive"):
    """
    Select edges correlated with y using a p-value threshold.
    edge_type: "positive", "negative", or "both"
    Returns mask of selected features.
    """
    n_edges = X.shape[1]
    corr_vals = np.zeros(n_edges)
    p_vals = np.zeros(n_edges)

    y_flat = y.squeeze()
    
    # Compute correlation for each edge with y
    for i in range(n_edges):
        r, p = pearsonr(X[:, i], y_flat)
        corr_vals[i] = r
        p_vals[i] = p

    # Feature selection logic
    if edge_type == "positive":
        mask = (p_vals < thresh) & (corr_vals > 0)
    elif edge_type == "negative":
        mask = (p_vals < thresh) & (corr_vals < 0)
    elif edge_type == "both":
        mask = p_vals < thresh
    else:
        raise ValueError("edge_type must be 'positive', 'negative', or 'both'")
    
    return mask, corr_vals[mask]

def cpm_method(X, y, thresh=0.01, edge_type="positive"):
    """
    CPM Workflow with edge_type option.
    X: (n_samples, n_edge_features) - Connectome edges
    y: (n_samples, 1) - Metric scores
    Returns:
        ns: (n_samples, 1) - Network strength for each subject
        mask: selection mask array (n_edge_features,)
    """
    completeMask, _ = cpm_feature_selection(X, y, thresh, edge_type)

    subjectFeatures = []
    # leave one out
    IDs = np.arange(X.shape[0])
    for sub in tqdm(IDs):
        train_index = np.where(IDs != sub)[0]
        test_index = np.where(IDs == sub)[0]
        trainFeatures, testFeatures = X[train_index], X[test_index]
        trainScore, testScore = y[train_index], y[test_index]

        mask, _ = cpm_feature_selection(trainFeatures, trainScore, thresh, edge_type)
        feature = np.sum(testFeatures[:, mask], axis=1).reshape(-1, 1)
        subjectFeatures.append(feature)
    subjectFeatures = np.concatenate(subjectFeatures, axis=0)

    return subjectFeatures, completeMask

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

    workers = 8
    type = "rest"
    subject_class = "remove_unknown"
    score_amount = 3
    modescore = "mocaz"
    df = madc_import(filename=madc_file)
    score_db = score_import(filename=score_file)
    files = np.array(get_files(path, score_db, type=type, subject_class=subject_class))

    dataset = RSFMRI_DATALOADER(files, transforms=None, database=df, score_database=score_db)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=False, num_workers=workers)

    subjectFeatures, Scores, masks = CPM(loader, score_amount=3, mode="mocaz", edge_type="negative", thresh=0.01)
    print(f"Feature matrix shape: {subjectFeatures.shape}")
    print(f"Score matrix shape: {Scores.shape}")

    # save subjectFeatures, Scores, and masks for later use
    np.save('/home/javier/weights/cpm-_subjectFeatures.npy', subjectFeatures)
    np.save('/home/javier/weights/cpm-_Scores.npy', Scores)
    np.save('/home/javier/weights/cpm-_masks.npy', masks)