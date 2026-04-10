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
from NeuroMamba.models.PatchTST import PatchTST
from sklearn.kernel_ridge import KernelRidge
from tqdm import tqdm
import glob
from NeuroMamba.utils.adni import ADNI_LOADER, adni_import, adni_collate

def train(model, trainloader, optimizer, scheduler, epochs=100):
    best_val = 100000
    for epoch in range(epochs):
        running_loss = 0.
        model.train()
        for idx, (data,info) in enumerate(trainloader):
            scores = mapScores(info, num=score_amount, mode=score_mode).to("cuda", dtype=torch.float32)
            inputs = data.to("cuda", dtype=torch.float32)
            optimizer.zero_grad()
            predictedScores = model(inputs)
            loss = criterion(predictedScores, scores)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(trainloader.dataset)
        if scheduler is not None:
            scheduler.step()

        print("Epoch ", epoch, " Train Loss: ", train_loss)

        if epoch == epochs-1:
            torch.save(model.state_dict(), os.path.join(f'/home/javier/weights/loo/patchtst/all.pth'))
    
    return None

def criterion(predictedScores, scores):
    fidelity = torch.nn.functional.mse_loss(predictedScores, scores)
    return fidelity

def evaluate(testloader):
    model = PatchTST(n_layers = n_layers, n_heads=n_heads, patch_length=patch_length, patch_stride=patch_stride, d_model = d_model, ffn_dim=ffn_dim, dropout=dropout, score_amount=score_amount).to("cuda")
    model.load_state_dict(torch.load(os.path.join(f'/home/javier/weights/loo/patchtst/all.pth')))
    model.eval()
    realScores = []
    predictedScores = []

    for idx, (data,info) in tqdm(enumerate(testloader), total=len(testloader)):
        data, info = adni_collate(data, info, file_mode=file_mode)
        real = mapScores(info, num=score_amount, mode=score_mode).cpu().numpy()
        inputs = data.to("cuda", dtype=torch.float32)
        for m in range(inputs.shape[0]):
            pred = None
            with torch.no_grad():
                input = inputs[m].unsqueeze(0)
                # change input of size [1, 192, 272] to size [1, 570, 272] by interpolation
                input = torch.nn.functional.interpolate(input.permute(0,2,1), size=570, mode='linear', align_corners=False).permute(0,2,1)
                # run through model
                pred = model(input)
                pred = pred[:,0].squeeze().cpu().numpy()

            realScores.append(real[m])
            predictedScores.append(pred)

    realScores = np.array(realScores).squeeze()
    predictedScores = np.array(predictedScores).squeeze()

    return realScores, predictedScores


if __name__ == "__main__":

    mode = "home"
    if mode == "home":
        experiment_dir = "/home/javier/Desktop/DeepScore/experiments/adni/zeroshot/"
        path = "/home/javier/madc/"
        madc_file = "/home/javier/Desktop/DeepScore/madc_complete.csv"
        score_file = "/home/javier/Desktop/DeepScore/scores.csv"
    elif mode == "server":
        experiment_dir = "/home/javiersc/DeepScore/experiments/adni/zeroshot/"
        path = "/home/javiersc/madc/"
        madc_file = "/home/javiersc/DeepScore/madc_complete.csv"
        score_file = "/home/javiersc/DeepScore/scores.csv"

    # params
    workers = 8
    df = adni_import()
    score_amount = 1
    score_mode = "mocaz"
    file_mode="all"
    # model params
    n_layers = 2
    n_heads = 4
    patch_length=16
    patch_stride=8
    d_model = 16
    ffn_dim = 128
    dropout=0.2
    
    lr = 1e-4
    epochs = 25
    batch_size = 32
    batch_factor = batch_size / 1
    lr = lr * np.sqrt(batch_factor)
    # madc
    madc_df = madc_import(filename=madc_file)
    madc_score = score_import(filename=score_file)
    madc_files = np.array(get_files(path, madc_df, type="rest", subject_class="remove_unknown"))
    madc_dataset = RSFMRI_DATALOADER(madc_files, transforms=None, database=madc_df, score_database=madc_score)
    madc_loader = DataLoader(madc_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers)
    # training
    scheduler = None
    model = PatchTST(n_layers = n_layers, n_heads=n_heads, patch_length=patch_length, patch_stride=patch_stride, d_model = d_model, ffn_dim=ffn_dim, dropout=dropout, score_amount=score_amount).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.0, 0.99))
    print("Training on MADC data...")
    train(model, madc_loader, optimizer, scheduler, epochs=epochs)
    # init folds
    files = sorted(glob.glob('/home/javier/adni/subjects/*'))
    dataset = ADNI_LOADER(files, transforms=None, database=df, file_mode=file_mode)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=workers)
    # LOO evaluation
    print("Evaluating on ADNI data...")
    real, pred = evaluate(loader)
    print(real.shape, pred.shape)
    # pearson correlation
    pearson_corr = stats.pearsonr(real, pred)
    pval = pearson_corr.pvalue
    pearson = pearson_corr.statistic
    print("Categories: [MoCA]")
    print("Final Pearson: ", pearson)
    print("P-values: ", pval)
