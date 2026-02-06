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
from tqdm import tqdm
from NeuroMamba.utils.adni import ADNI_LOADER, adni_import, adni_collate
import glob


if __name__ == "__main__":
    experiment_dir = "/home/javier/Desktop/DeepScore/experiments/adni/end2end/"
    # params
    workers = 8
    df = adni_import()
    score_amount = 1
    score_mode = "mocaz"
    file_mode="all"
    batch_size = 1
    # init folds
    files = np.array(sorted(glob.glob('/home/javier/adni/subjects/*')))
    dataset = ADNI_LOADER(files, transforms=None, database=df, file_mode="all")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=workers)

    # init dataframe
    adni = pd.DataFrame()

    for idx, (data,info) in enumerate(loader):
        data, info = adni_collate(data, info, file_mode=file_mode)
        for i in range(len(info['subjectID'])):
            entry = {
                'subjectID': info['subjectID'][i],
                'filename': info['filename'][i],
                'label': info['label'][i],
                'mocaz': info['cognitionz'][i].item(),
            }
            adni = pd.concat([adni, pd.DataFrame([entry])], ignore_index=True)

    print("Number of unique subjects", adni['subjectID'].nunique())
    print("Number of unique subjects for cn class:", adni[adni['label'] == 'cn']['subjectID'].nunique())
    print("Number of unique subjects for amci class:", adni[adni['label'] == 'amci']['subjectID'].nunique())
    print("Number of unique subjects for dat class:", adni[adni['label'] == 'dat']['subjectID'].nunique())

    print("Number of total sessions", adni['subjectID'].count())
    print("Number of total sessions for cn class:", adni[adni['label'] == 'cn']['subjectID'].count())
    print("Number of total sessions for amci class:", adni[adni['label'] == 'amci']['subjectID'].count())
    print("Number of total sessions for dat class:", adni[adni['label'] == 'dat']['subjectID'].count())