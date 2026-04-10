import torch
import torch.nn as nn
import math
from torchinfo import summary
from NeuroMamba.utils.fmri import *
from NeuroMamba.utils.dataloaders import get_files, RSFMRI_DATALOADER, get_dataframe_entry
from NeuroMamba.utils.fmri import madc_import, score_import, mapScores, mapClasses
from torch.utils.data import Dataset, DataLoader
from NeuroMamba.utils.adni import adni_collate

def FCM(loader, score_amount=3, mode="mocaz"):
    SubjectFeatures = []
    Scores = []

    for idx, (data,info) in enumerate(loader):
        x = data.to(device="cpu")
        x = generateStaticFC(x)
        x = matrix2vec(x)
        SubjectFeatures.append(x.numpy())
        score = mapScores(info, num=score_amount, mode=mode)
        Scores.append(score)

    Scores = np.array(Scores).squeeze()
    SubjectFeatures = np.array(SubjectFeatures).squeeze()

    return SubjectFeatures, Scores

def FCM_ADNI(loader, score_amount=3, mode="mocaz", file_mode="all"):
    SubjectFeatures = []
    Scores = []

    for idx, (data,info) in enumerate(loader):
        data, info = adni_collate(data, info, file_mode=file_mode)
        x = data.to(device="cpu")
        x = generateStaticFC(x)
        x = matrix2vec(x)
        SubjectFeatures.append(x.numpy())
        score = mapScores(info, num=score_amount, mode=mode)
        Scores.append(score)

    Scores = np.concatenate(Scores).squeeze()
    SubjectFeatures = np.concatenate(SubjectFeatures).squeeze()

    if len(SubjectFeatures.shape) == 1:
        SubjectFeatures = SubjectFeatures[None, :]

    return SubjectFeatures, np.atleast_1d(Scores)

if __name__ == "__main__":
    mode = "home"
    if mode == "home":
        experiment_dir = "/home/javier/Desktop/DeepScore/experiments/leaveoneout/"
        path = "/home/javier/madc/"
        madc_file = "/home/javier/Desktop/DeepScore/madc_complete.csv"
        score_file = "/home/javier/Desktop/DeepScore/scores.csv"
    elif mode == "server":
        experiment_dir = "/home/javiersc/DeepScore/experiments/leaveoneout/"
        path = "/home/javiersc/madc/"
        madc_file = "/home/javiersc/DeepScore/madc_complete.csv"
        score_file = "/home/javiersc/DeepScore/scores.csv"

    workers = 8
    type = "rest"
    subject_class = "remove_unknown"
    score_amount = 3
    modescore = "mocaz"
    df = madc_import(filename=madc_file)
    score_db = score_import(filename=score_file)
    files = np.array(get_files(path, df, type=type, subject_class=subject_class))

    dataset = RSFMRI_DATALOADER(files, transforms=None, database=df, score_database=score_db)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=False, num_workers=workers)

    subjectFeatures, Scores, Labels = FCM(loader, score_amount=score_amount, mode=modescore)
    print(f"Feature matrix shape: {subjectFeatures.shape}")
    print(f"Score matrix shape: {Scores.shape}")
    print(f"Labels shape: {len(Labels)}")
    print(Labels)