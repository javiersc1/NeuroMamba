from pathlib import Path
import numpy as np
import torch
import os
import glob
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from NeuroMamba.utils.fmri import madc_import, score_import
import time

def get_dataframe_entry(df, subjectID, filterType="mocatots"):
    # find the session in the dataframe closest to the MRI date

    # exam session
    exam = subjectID.split('e')[-1]
    # madc specific ID slightly different string for master table
    madcID = subjectID.split('e')[0]
    madcID = madcID.split('umm')[-1]
    madcID = "UM000"+madcID

    # extract subject specific sessions
    sessions = df.loc[df['ptid'] == madcID]

    # filter sessions by exam session
    sessions = sessions.loc[sessions['exam_session'] == exam]

    # Convert dates from string to datetime
    sessions['form_date'] = pd.to_datetime(sessions['form_date'], format='%d-%b-%Y')
    sessions['mri_date'] = pd.to_datetime(sessions['mri_date'], format='%d-%b-%Y')

    # filter for clean rows mocatots no nan
    sessions = sessions.dropna(subset=[filterType])

    # Calculate the absolute difference between form_date and mri_date
    sessions['date_difference'] = (sessions['form_date'] - sessions['mri_date']).abs()

    # Find the row with the minimum date difference
    session = sessions.loc[sessions['date_difference'].idxmin()] 

    return session

class RSFMRI_DATALOADER(Dataset):
    def __init__(self, path, transforms=None, database=None, score_database=None):
        # load all nii handle in a list
        self.image_paths = [image_path for image_path in path]
        self.transforms = transforms
        self.df = database
        self.score_df = score_database


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # get numpy matrix
        numpy_image = self.image_paths[idx]
        data = torch.from_numpy(np.load(numpy_image))
        # filename
        filename = str(self.image_paths[idx])
        # subject ID
        subjectID = os.path.basename(os.path.dirname(filename))
        # get database entry
        #session = get_dataframe_entry(self.df, subjectID)

        # get score data
        sub_scores = self.score_df.loc[self.score_df['subjectID'] == subjectID]

        avg_memory = sub_scores['composite_memory'].values
        avg_language = sub_scores['composite_language'].values
        cognitionz = sub_scores['cognition'].values
        label = sub_scores['label_alternative'].values[0]
        race = sub_scores['race'].values[0]
        gender = sub_scores['gender'].values[0]

        # compile dict with info that we want
        info = { 'subjectID': subjectID,
                'filename': filename,
                'label': label,
                'race': race,
                'gender': gender,                    
                'cognitionz': cognitionz,
                'avg_memory': avg_memory,
                'avg_language': avg_language,
                }

        return data, info

    
def filter_files(files, df, filterType="remove_unknown"):
    # filter files based on subject class e.g. cn only
    filtered_files = []
    for file in files:
        flag = 0
        # get subjectID from file
        subjectID = os.path.basename(os.path.dirname(file))
        
        # get database entry
        #session = get_dataframe_entry(df, subjectID)
        #session = df[df['subjectID'] == subjectID]

        #check if subjectID exists in dataframe
        session = df.loc[df['subjectID'] == subjectID]
        if session.empty:
            flag = 1
        else:
            session = session.iloc[0]

        if flag == 0:
            if filterType == "remove_unknown" and session['label_alternative'] != "unknown" and flag == 0:
                filtered_files.append(file)

            if  filterType == "cn" and session['label_alternative'] == "cn" and flag == 0:
                filtered_files.append(file)

            if filterType == "dat" and session['label_alternative'] == "dat" and flag == 0:
                filtered_files.append(file)

            if filterType == "mci" and (session['label_alternative'] == "mci" or session['label_alternative'] == "amci" or session['label_alternative'] == "namci") and flag == 0:
                filtered_files.append(file)

            if filterType == "2class":
                if (session['label_alternative'] == "cn" or session['label_alternative'] == "dat") and flag == 0:
                    filtered_files.append(file)
                
            if filterType == "multiclass":
                if (session['label_alternative'] == "cn" or session['label_alternative'] == "dat" or session['label_alternative'] == "amci" or session['label_alternative'] == "namci") and flag == 0:
                    filtered_files.append(file)

        
    return filtered_files
    
def get_files(path, database, type="rest", subject_class="all"):
    # get all files in madc folder, only resting state files supported
    # and filter by subject class e.g. cn, dat, mci, remove_unknown
    files = []
    if type == "rest" and subject_class == "all":
        files = sorted(glob.glob(path+'*/restingstate.npy'))
    elif type == "rest" and subject_class != "all":
        files = sorted(glob.glob(path+'*/restingstate.npy'))
        files = filter_files(files, database, filterType=subject_class)

    return files

def divide_files(files, mode="val", split=0.7):
    # divide files into training and validation sets
    split_index = int(len(files) * split)
    new_files = []
    if mode == "val":
        new_files = files[split_index:]
    if mode == "train":
        new_files = files[:split_index]
    
    return new_files

def get_rsfmri_dataloader(path="/home/javiersc/madc/", database=None, score_db=None, mode="val", subject_class="cn", batch_size=1, split=0.7):
    workers = 8
    files = get_files(path, database, type="rest", subject_class=subject_class)
    files = divide_files(files, mode=mode, split = split)
    print("class: "+subject_class+" and "+str(len(files))+" sessions.")
    # dataset
    dataset = RSFMRI_DATALOADER(files, transforms=None, database=database, score_database=score_db)
    # generate dataloader
    shuffleMode, dropMode = False, False
    if mode == "train":
        shuffleMode = True
        dropMode = True
    elif mode == "val":
        shuffleMode = False
        dropMode = False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffleMode, drop_last=dropMode, num_workers=workers)
    return dataloader

if __name__ == '__main__':
    df = madc_import("/home/javier/Desktop/DeepScore/madc_complete.csv")
    score_db = score_import("/home/javier/Desktop/DeepScore/scores.csv")
    batch_size = 1
    subject_class = "all" # options: "all", "cn", "dat", "mci", "remove_unknown"
    split = 1.0 # 1.0 means everything in training set
    mode = "train" # val or train

    # dataloader = get_dataloader(path="/home/javier/madc/", database=df, type="rest", mode=mode, subject_class=subject_class, batch_size=batch_size, split=split)
    # rows = []
    # print("------IMAGE CHARACTERISTICS--------")
    # for idx, (data,info) in enumerate(dataloader):
    #     row = {'subjectID': info['subjectID'][0], 
    #            'race': info['race'][0],
    #            'gender': info['gender'].item(),
    #            'label': info['label'][0],
    #            'mocatots': info['mocatots'].item(), 
    #            'udsbentd': info['udsbentd'].item(), 
    #            'craftdvr': info['craftdvr'].item(), 
    #            'craftdre': info['craftdre'].item(), 
    #            'craftvrs': info['craftvrs'].item(), 
    #            'animals_c2': info['animals_c2'].item(),
    #            'veg_c2': info['veg_c2'].item(),
    #            'minttots': info['minttots'].item()
    #            }
    #     rows.append(row)

    # score_df = pd.DataFrame(rows)
    # score_df.to_csv("scores.csv", index=False)
    #     # print('Subject: '+info['subjectID'][0])
    #     # print('MOCA: '+str(info['mocatots'].item())  + ', UDSBENTD: '+str(info['udsbentd'].item()) + ', CRAFTDVR: '+str(info['craftdvr'].item())
    #     #       + ', CRAFTDRE: '+str(info['craftdre'].item()) + ', CRAFTVRS: '+str(info['craftvrs'].item()) + ', ANIMALS_C2: '+str(info['animals_c2'].item())
    #     #       + ', VEG_C2: '+str(info['veg_c2'].item()) + ', MINTTOTS: '+str(info['minttots'].item()))

    dataloader = get_rsfmri_dataloader(path="/home/javier/madc/", database=df, score_db=score_db, mode=mode, subject_class=subject_class, batch_size=batch_size, split=split)
    for idx, (data,info) in enumerate(dataloader):
        print('Subject: '+info['subjectID'][0])
        print('MOCAz: '+str(info['cognition'].item())  + ', avg_memoryz: '+str(info['avg_memory'].item()) + ', avg_languagez: '+str(info['avg_language'].item())
              + ', avg_learningz: '+str(info['avg_learning'].item()))
        
