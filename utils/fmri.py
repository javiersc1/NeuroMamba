import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd

def mapScores(info, num=1, mode="mocaraw"):
    if num == 1 and mode=="mocaraw":
        return info['cognition']
    elif num == 1 and mode=="mocaz":
        return info['cognitionz']
    elif num ==1 and mode=="language":
        return info['avg_language']
    elif num == 1 and mode=="memory":
        return info['avg_memory']
    elif num == 3 and mode=="mocaz":
        all_scores = torch.cat([info['cognitionz'], info['avg_memory'], info['avg_language']], dim=-1)
        return all_scores
    elif num == 3 and mode=="mocaraw":
        all_scores = torch.cat([info['cognition'], info['avg_memory'], info['avg_language']], dim=-1)
        return all_scores
    elif num == 4:
        all_scores = torch.cat([info['cognitionz'], info['avg_memory'], info['avg_language'], info['avg_learning']], dim=-1)
        return all_scores


def mapClasses(y, n_classes):
    classes = -1.0 + np.zeros(len(y))
    idx = 0
    if n_classes == 2:
        for i in y:
            if i == "cn":
                classes[idx] = 0
            elif i == "dat":
                classes[idx] = 1
            elif i == "amci":
                classes[idx] = 1
            idx = idx + 1
    elif n_classes == 3:
        for i in y:
            if i == "cn":
                classes[idx] = 0
            elif i == "dat":
                classes[idx] = 2
            elif i == "amci":
                classes[idx] = 1
            idx = idx + 1
    elif n_classes == 4:
        for i in y:
            if i == "cn":
                classes[idx] = 0
            elif i == "dat":
                classes[idx] = 3
            elif i == "amci":
                classes[idx] = 2
            elif i == "namci":
                classes[idx] = 1
            idx = idx + 1
    return classes

def generateDynamicFC(X, TR=1.0, TW=30.0):
    # X: (B,L,C)
    # Y: (B,L,C,C)
    B,L,C = X.size()
    window = int(TW // TR)
    if window % 2 != 0:
        window += 1
    X = torch.permute(X, (0,2,1))
    X = F.pad(X, (window, window), "reflect")
    y = torch.zeros(B,L,C,C,device=X.device)
    for b in range(B):
        for l in range(L):
            y[b,l,:,:] = torch.corrcoef(X[b,:,l:l+window])
    return y

def generateStaticFC(X):
    # X: (B,L,C)
    # Y: (B,C,C)
    B,L,C = X.size()
    X = torch.permute(X, (0,2,1))
    y = torch.zeros(B,C,C,device=X.device)
    for b in range(B):
        y[b,:,:] = torch.corrcoef(X[b,:,:])
    return y

def matrix2vec(F):
    if F.dim() == 3:
        # F: (B, C, C) 
        # Y: (B, 0.5*C*C-1)
        B, C, C = F.size()
        idx = torch.triu_indices(C, C, 1)
        y = torch.zeros(B,C*(C-1)//2, device=F.device)
        y = F[:, idx[0,:], idx[1,:]]
    elif F.dim() == 4:
        # F: (B, L, C, C) 
        # Y: (B, L, 0.5*C*C-1)
        B, L, C, C = F.size()
        y = torch.zeros(B,L,C*(C-1)//2, device=F.device)
        idx = torch.triu_indices(C, C, 1)
        y = F[:, :, idx[0,:], idx[1,:]]

    return y

def vec2matrix(x, C:int, onlyHalf=False):
    if x.dim() == 2:
        # X: (B,D)
        # F: (B,C,C)
        B,D = x.size()
        F = torch.zeros(B,C,C,device=x.device)
        idx = torch.triu_indices(C,C,1)
        F[:,idx[0,:], idx[1,:]] = x
        if onlyHalf==True:
            return F
        else:
            F = F + torch.transpose(F,1,2) #+ torch.eye(C, device=x.device)
            return F
    elif x.dim() == 3:
        # X: (B,L,D)
        # F: (B,L,C,C)
        B,L,D = x.size()
        F = torch.zeros(B,L,C,C,device=x.device)
        idx = torch.triu_indices(C,C,1)
        F[:, :, idx[0,:], idx[1,:]] = x
        if onlyHalf==True:
            return F
        else:
            F = F + torch.transpose(F,2,3) #+ torch.eye(C, device=x.device)
            return F
    

def power_import(filename="/home/javier/Desktop/DeepScore/power_atlas.csv"):
    df = pd.read_csv(filename)
    return df

def madc_import(filename="/home/javier/Desktop/DeepScore/madc_complete.csv"):
    df = pd.read_csv(filename)
    return df

def score_import(filename="/home/javier/Desktop/DeepScore/scores.csv"):
    df = pd.read_csv(filename)
    return df

if __name__ == "__main__":
    df = power_import()
    print(df)

    # x = torch.randn((2,100,4))
    # y = generateDynamicFC(x)
    # y_vec = matrix2vec(y)
    # y_result = vec2matrix(y_vec, 4, onlyHalf=False)
    # print(y[0,50,:,:])
    # print(y_result[0,50,:,:])