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
import glob
from NeuroMamba.utils.adni import ADNI_LOADER, adni_import, adni_collate

def finetune(model, trainloader, optimizer, scheduler, epochs=100):
    best_val = 100000
    for epoch in range(epochs):
        running_loss = 0.
        model.train()
        for idx, (data,info) in enumerate(trainloader):
            data, info = adni_collate(data, info, file_mode=file_mode)
            scores = mapScores(info, num=score_amount, mode=score_mode).to("cuda", dtype=torch.float32).unsqueeze(1)
            inputs = data.to("cuda", dtype=torch.float32)
            optimizer.zero_grad()
            predictedScores, latents = model(inputs)
            loss = criterion(predictedScores, latents, scores)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()

        
        train_loss = running_loss / len(trainloader.dataset)
        if scheduler is not None:
            scheduler.step()

        print("Epoch ", epoch, " Train Loss: ", train_loss)

        if epoch==epochs-1:
            torch.save(model.state_dict(), os.path.join(f'/home/javier/weights/loo/neuromamba/fewshot.pth'))
    
    return None

def criterion(predictedScores, latents, scores):
    fidelity = torch.nn.functional.mse_loss(predictedScores, scores)
    reg = torch.linalg.vector_norm(latents, ord=1, dim=-1).mean()
    return fidelity + lam * reg

def evaluate(testloader):
    model = NeuroMamba(n_layers=n_layers, state_dim=state_dim, num_variables=272, score_amount=score_amount_train, dt_max=dt_max).to("cuda")
    model.load_state_dict(torch.load(os.path.join(f'/home/javier/weights/loo/neuromamba/fewshot.pth')))
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
                pred, _ = model(input)
                pred = pred[:,0].squeeze().cpu().numpy()
            realScores.append(real[m])
            predictedScores.append(pred)

    realScores = np.array(realScores).squeeze()
    predictedScores = np.array(predictedScores).squeeze()

    return realScores, predictedScores


if __name__ == "__main__":

    mode = "home"
    if mode == "home":
        experiment_dir = "/home/javier/Desktop/DeepScore/experiments/adni/fewshot/"
        path = "/home/javier/madc/"
        madc_file = "/home/javier/Desktop/DeepScore/madc_complete.csv"
        score_file = "/home/javier/Desktop/DeepScore/scores.csv"
    elif mode == "server":
        experiment_dir = "/home/javiersc/DeepScore/experiments/adni/fewshot/"
        path = "/home/javiersc/madc/"
        madc_file = "/home/javiersc/DeepScore/madc_complete.csv"
        score_file = "/home/javiersc/DeepScore/scores.csv"

    # params
    workers = 8
    df = adni_import()
    score_amount = 1
    score_mode = "mocaz"
    score_amount_train = 1  
    file_mode="all"
    # model params
    n_layers = 12
    state_dim = 32
    dt_max = 1.0
    lr = 1e-4
    lam = 0.01
    finetune_epochs = 5
    batch_size = 1

    # FEW SHOT LEARNING ON ADNI DATA
    # get all adni files
    files = sorted(glob.glob('/home/javier/adni/subjects/*'))
    labels = []
    # get labels for files list
    for i in range(len(files)):
        subject_id = files[i].split('/')[-1]
        # where column filename matches subject_id
        df_entry = df[df['subject'] == subject_id]
        # get label from df_entry
        label = df_entry['label'].iloc[0]
        labels.append(label)
    labels = np.array(labels)
    # take five subjects from each class for fewshot training
    fewshot_files = []
    for cls in np.unique(labels):
        cls_indices = np.where(labels == cls)[0]
        # randomly select 5 indices with seed
        np.random.seed(2025)
        np.random.shuffle(cls_indices)
        selected_indices = cls_indices[:5]
        print(f"Class {cls}: selected indices {selected_indices}")
        fewshot_files.extend([files[i] for i in selected_indices])
    # form dataset and dataloader
    fewshot_dataset = ADNI_LOADER(fewshot_files, transforms=None, database=df, file_mode=file_mode)
    fewshotloader = DataLoader(fewshot_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers)
    model = NeuroMamba(n_layers=n_layers, state_dim=state_dim, num_variables=272, score_amount=score_amount_train, dt_max=dt_max).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-08, betas=(0.0, 0.95))
    scheduler = None
    model.load_state_dict(torch.load(os.path.join(f'/home/javier/weights/loo/neuromamba/all.pth')))
    print("Fine-tuning on ADNI data...")
    finetune(model, fewshotloader, optimizer, scheduler, epochs=finetune_epochs)

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
