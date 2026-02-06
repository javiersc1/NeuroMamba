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
from NeuroMamba.models.NeuroMamba import NeuroMamba
from sklearn.kernel_ridge import KernelRidge
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

def train(model, trainloader, valloader, optimizer, scheduler, epochs=100, fold=0):
    best_val = 100000
    for epoch in range(epochs):
        running_loss = 0.
        model.train()
        for idx, (data,info) in enumerate(trainloader):
            scores = mapScores(info, num=score_amount, mode=score_mode).to("cuda", dtype=torch.float32)
            #labels = mapClasses(info['label'], 2)
            #labels = torch.tensor(labels, device="cuda", dtype=torch.float32).unsqueeze(-1)
            inputs = data.to("cuda", dtype=torch.float32)
            optimizer.zero_grad()
            pred, latents = model(inputs)
            loss = criterion(pred, latents, scores)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        
        train_loss = running_loss / len(trainloader.dataset)
        if scheduler is not None:
            scheduler.step()

        running_loss = 0.
        totalScores = []
        totalPredictedScores = []
        model.eval()
        for idx, (data,info) in enumerate(valloader):
            scores = mapScores(info, num=score_amount, mode=score_mode).to("cuda", dtype=torch.float32)
            #labels = mapClasses(info['label'], 2)
            #labels = torch.tensor(labels, device="cuda", dtype=torch.float32).unsqueeze(-1)
            inputs = data.to("cuda", dtype=torch.float32)
            with torch.no_grad():
                pred, latents = model(inputs)
                loss = criterion(pred, latents, scores)
                running_loss += loss.item()
        val_loss = running_loss / len(valloader.dataset)

        print("")
        print(f"Epoch {epoch+1}/{epochs} - Trial {fold} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")

        if epoch in [epochs-1]:
            torch.save(model.state_dict(), os.path.join(f'/home/javier/weights/loo/neuromamba2/val_{fold}.pth'))
    
    return None

def evaluate(testloader, fold):
    model = NeuroMamba(n_layers=n_layers, state_dim=state_dim, num_variables=272, score_amount=score_amount).to("cuda")
    model.load_state_dict(torch.load(os.path.join(f'/home/javier/weights/loo/neuromamba2/val_{fold}.pth')))
    model.eval()
    predicted = []
    true = []

    for idx, (data,info) in enumerate(testloader):
        scores = mapScores(info, num=score_amount, mode=score_mode).to("cuda", dtype=torch.float32)
        #labels = mapClasses(info['label'], 2)
        #labels = torch.tensor(labels, device="cuda", dtype=torch.float32).unsqueeze(-1)
        inputs = data.to("cuda", dtype=torch.float32)
        with torch.no_grad():
            pred, _ = model(inputs)
            predicted.append(pred.cpu().numpy())
            true.append(scores.cpu().numpy())
    
    predicted = np.concatenate(np.array(predicted))
    true = np.concatenate(np.array(true))

    return predicted, true

def criterion(predictedScores, latents, scores):
    fidelity = torch.nn.functional.mse_loss(predictedScores, scores)
    reg = torch.linalg.vector_norm(latents, ord=1, dim=-1).mean()
    return fidelity + lam * reg


def leaveoneout(files):
    subs = [file.split('/')[-2] for file in files]
    sessions = np.array([get_dataframe_entry(df, subjectID, filterType="mocatots")['label'] for subjectID in subs])
    labelsKF = mapClasses(sessions, 2).astype(int)

    kf = StratifiedKFold(n_splits=4, random_state=42, shuffle=True)
    predicted = []
    true = []
    trial = 1

    for train_index, test_index in tqdm(kf.split(files, labelsKF), total=kf.get_n_splits(files, labelsKF), desc="CV"):
        train_files = files[train_index]
        test_file = files[test_index]
        # create datasets
        train_dataset = RSFMRI_DATALOADER(train_files, transforms=None, database=df, score_database=score_db)
        test_dataset = RSFMRI_DATALOADER(test_file, transforms=None, database=df, score_database=score_db)
        # create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=workers)
        # initialize model
        torch.manual_seed(42)
        model = NeuroMamba(n_layers=n_layers, state_dim=state_dim, num_variables=272, score_amount=score_amount).to("cuda")
        # train model
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-08, betas=(0.0, 0.95))
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scheduler = None
        train(model, train_loader, test_loader, optimizer, scheduler, epochs=epochs, fold=trial)
        # evaluate model
        pred, real = evaluate(test_loader, fold=trial)
        predicted.append(pred)
        true.append(real)

        trial += 1

    predicted = np.concatenate(predicted)
    true =  np.concatenate(true)
    #predictedProb = np.array(predictedProb).flatten()
    #labels = np.array(labels).flatten()

    return predicted, true


if __name__ == "__main__":

    mode = "home"
    if mode == "home":
        experiment_dir = "/home/javier/Desktop/DeepScore/experiments/validate/"
        path = "/home/javier/madc/"
        madc_file = "/home/javier/Desktop/DeepScore/madc_complete.csv"
        score_file = "/home/javier/Desktop/DeepScore/scores.csv"
    elif mode == "server":
        experiment_dir = "/home/javiersc/DeepScore/experiments/validate/"
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
    score_mode = "mocaz"
    # model params
    n_layers = 12
    state_dim = 32
    lam = 0.01
    lr = 1e-4
    epochs = 25
    batch_size = 32

    batch_factor = batch_size / 1
    lr = lr * np.sqrt(batch_factor)
    # init folds
    files = np.array(get_files(path, df, type=type, subject_class=subject_class))
    # LOO evaluation
    pred, real = leaveoneout(files)
    # get results
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
    print("Final Pearson: ", pearson)
    print("P-values: ", pvals)
