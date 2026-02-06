from pathlib import Path
import numpy as np
import torch
import os
import glob
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import time

def adni_collate(data, info, file_mode="all"):
    if file_mode == "train":
        return data, info
    if file_mode == "all":
        n_batch, n_scans, n_time, n_regions = data.shape
        # merge batch and scans
        data = data[0,:,:,:] #data.view(n_batch * n_scans, n_time, n_regions)
        # merge info
        merged_info = {
            'subjectID': [i[0] for i in info['subjectID']],
            'filename': [i[0] for i in info['filename']],
            'label': [i[0] for i in info['label']],
            'cognition': info['cognition'].view(-1),
            'cognitionz': info['cognitionz'].view(-1),
        }
        return data, merged_info


def adni_import(filename="/home/javier/Desktop/DeepScore/adni.csv"):
    df = pd.read_csv(filename)
    return df

class ADNI_LOADER(Dataset):
    def __init__(self, path, transforms=None, database=None, file_mode="train"):
        # load all nii handle in a list
        self.image_paths = [image_path for image_path in path]
        self.transforms = transforms
        self.df = database
        self.file_mode = file_mode


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # get numpy matrix
        filename, data, label, mocaz, mocatots, subjectID = None, None, None, None, None, None
        n_time_total = 192 #1000
        n_regions = 272

        if self.file_mode == "train":
            scans = glob.glob(self.image_paths[idx]+"/*.npy")
            n_scans = len(scans)
            chosen_scan = np.random.randint(0, n_scans)

            numpy_image = scans[chosen_scan]
            data = torch.from_numpy(np.load(numpy_image))
            n_time = data.shape[0]

            # pad data
            # pad_amount = n_time_total - n_time
            # pad_tensor = torch.zeros((pad_amount, n_regions), dtype=data.dtype)
            # data = torch.cat((data, pad_tensor), dim=0)

            # keep last 193 time points
            data = data[-n_time_total:, :]

            # get entry
            filename = str(numpy_image).split('/')[-1]
            session = self.df.loc[self.df['filename'] == filename]
            mocaz = session['mocaz'].values[0]
            mocatots = session['mocatots'].values[0]
            subjectID = session['subject'].values[0]
            label = session['label'].values[0]

        if self.file_mode == "all":
            scans = glob.glob(self.image_paths[idx]+"/*.npy")
            n_scans = len(scans)
            # empty data tensor
            data = torch.zeros((n_scans, n_time_total, n_regions), dtype=torch.float32)
            label = []
            mocaz = torch.zeros((n_scans,), dtype=torch.float32)
            mocatots = torch.zeros((n_scans,), dtype=torch.float32)
            filename = []
            subjectID = []

            for s_idx, scan in enumerate(scans):
                numpy_image = scan
                scan_data = torch.from_numpy(np.load(numpy_image))
                n_time = scan_data.shape[0]

                # pad time
                # pad_amount = n_time_total - n_time
                # pad_tensor = torch.zeros((pad_amount, n_regions), dtype=scan_data.dtype)
                # scan_data = torch.cat((scan_data, pad_tensor), dim=0)

                # keep last 193 time points
                scan_data = scan_data[-n_time_total:, :]

                # save data
                data[s_idx] = scan_data
                scan_file = str(numpy_image).split('/')[-1]
                filename.append(scan_file)
                scan_label = self.df.loc[self.df['filename'] == scan_file]['label'].values[0]
                label.append(scan_label)
                scan_mocaz = self.df.loc[self.df['filename'] == scan_file]['mocaz'].values[0]
                mocaz[s_idx] = scan_mocaz
                scan_mocatots = self.df.loc[self.df['filename'] == scan_file]['mocatots'].values[0]
                mocatots[s_idx] = scan_mocatots
                scan_subjectID = self.df.loc[self.df['filename'] == scan_file]['subject'].values[0]
                subjectID.append(scan_subjectID)
                

        
        # compile dict with info that we want
        info = { 'subjectID': subjectID,
                'filename': filename,
                'label': label,
                'cognition': mocatots,
                'cognitionz': mocaz,
                }

        return data, info
    
if __name__ == '__main__':
    df = adni_import()
    files = sorted(glob.glob('/home/javier/adni/subjects/*'))
    dataset = ADNI_LOADER(files, transforms=None, database=df, file_mode="all")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
    for idx, (data, info) in enumerate(dataloader):
        data, info = adni_collate(data, info, file_mode="all")
        print(data.shape)
        print(info['cognition'])
        print(info['label'])
        # print(info['subjectID'])
        # print(info['filename'])
        # print(info['cognitionz'])