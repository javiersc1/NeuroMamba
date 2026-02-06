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
from NeuroMamba.models.NeuroMamba import NeuroMambaCE
from sklearn.kernel_ridge import KernelRidge
from tqdm import tqdm

def train(model, trainloader, valloader, optimizer, scheduler, epochs=100, fold=0):
    best_val = 100000
    for epoch in range(epochs):
        running_loss = 0.
        model.train()
        for idx, (data,info) in enumerate(trainloader):
            scores = mapScores(info, num=score_amount, mode=score_mode).to("cuda", dtype=torch.float32)
            labels = mapClasses(info['label'], 2)
            labels = torch.tensor(labels, device="cuda", dtype=torch.float32).unsqueeze(-1)
            inputs = data.to("cuda", dtype=torch.float32)
            optimizer.zero_grad()
            logits, latents = model(inputs, scores)
            loss = criterion(logits, latents, labels)
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
            labels = mapClasses(info['label'], 2)
            labels = torch.tensor(labels, device="cuda", dtype=torch.float32).unsqueeze(-1)
            inputs = data.to("cuda", dtype=torch.float32)
            with torch.no_grad():
                logits, latents = model.evaluate(inputs, scores)
                loss = criterion(logits, latents, labels)
                running_loss += loss.item()
        val_loss = running_loss / len(valloader.dataset)

        #print("")
        #print(f"Epoch {epoch+1}/{epochs} - Subject {fold} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
        #print(f"Epoch {epoch} - Subject {fold} -  Logit: {logits.flatten().cpu().numpy()}")

        if epoch == epochs-1:
            torch.save(model.state_dict(), os.path.join(f'/home/javier/weights/loo/cross_entropy/{fold}.pth'))
    
    return None

def evaluate(testloader, fold):
    model = NeuroMambaCE(n_layers=n_layers, state_dim=state_dim, num_variables=272).to("cuda")
    model.load_state_dict(torch.load(os.path.join(f'/home/javier/weights/loo/cross_entropy/{fold}.pth')))
    model.eval()
    predictedProb = 0.
    label = 0

    for idx, (data,info) in enumerate(testloader):
        scores = mapScores(info, num=score_amount, mode=score_mode).to("cuda", dtype=torch.float32)
        labels = mapClasses(info['label'], 2)
        labels = torch.tensor(labels, device="cuda", dtype=torch.float32).unsqueeze(-1)
        inputs = data.to("cuda", dtype=torch.float32)
        with torch.no_grad():
            predictedProb, _ = model.evaluate(inputs, scores)
    
    predictedProb = predictedProb.cpu().numpy().flatten()
    label = labels.cpu().numpy().flatten()

    return predictedProb, label

def criterion(predictedLogits, latents, labels):
    fidelity = crossEntropy(predictedLogits, labels)
    reg = torch.linalg.vector_norm(latents, ord=1, dim=-1).mean()
    return fidelity + lam * reg

def leaveoneout(files):
    loo = LeaveOneOut()
    predictedProb = []
    labels = []
    trial = 1

    for train_index, test_index in tqdm(loo.split(files), total=loo.get_n_splits(files), desc="LOO CV"):
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
        model = NeuroMambaCE(n_layers=n_layers, state_dim=state_dim, num_variables=272).to("cuda")
        # train model
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.0, 0.99), weight_decay=weight_decay)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scheduler = None
        train(model, train_loader, test_loader, optimizer, scheduler, epochs=epochs, fold=trial)
        # evaluate model
        predictedLogit, label = evaluate(test_loader, fold=trial)
        print(f"Subject {trial} - Label: {label} - Predicted Probability: {predictedLogit}")
        predictedProb.append(predictedLogit)
        labels.append(label)

        trial += 1

    predictedProb = np.array(predictedProb).flatten()
    labels = np.array(labels).flatten()

    return predictedProb, labels


if __name__ == "__main__":

    mode = "home"
    if mode == "home":
        experiment_dir = "/home/javier/Desktop/DeepScore/experiments/crossentropy/"
        path = "/home/javier/madc/"
        madc_file = "/home/javier/Desktop/DeepScore/madc_complete.csv"
        score_file = "/home/javier/Desktop/DeepScore/scores.csv"
    elif mode == "server":
        experiment_dir = "/home/javiersc/DeepScore/experiments/crossentropy/"
        path = "/home/javiersc/madc/"
        madc_file = "/home/javiersc/DeepScore/madc_complete.csv"
        score_file = "/home/javiersc/DeepScore/scores.csv"

    # params
    workers = 8
    df = madc_import(filename=madc_file)
    score_db = score_import(filename=score_file)
    type = "rest"
    subject_class = "remove_unknown"
    score_amount = 1
    score_mode = "mocaz"
    # model params
    n_layers = 12
    state_dim = 32
    lr = 1e-4
    lam = 0.01
    weight_decay = 0.0
    epochs = 5
    batch_size = 32
    old_size = 1
    batch_factor = batch_size / old_size
    lr = lr * np.sqrt(batch_factor)
    #weights = torch.tensor([2.61], dtype=torch.float32).to("cuda")
    crossEntropy = torch.nn.BCEWithLogitsLoss(pos_weight=None, reduction='mean')
    # init folds
    files = np.array(get_files(path, df, type=type, subject_class=subject_class))
    # LOO evaluation
    predictedProb, labels = leaveoneout(files)
    # save predictions
    results_df = pd.DataFrame({'Predicted_Probability': predictedProb, 'Label': labels})
    results_df.to_csv(os.path.join(experiment_dir, 'neuromamba.csv'), index=False)
