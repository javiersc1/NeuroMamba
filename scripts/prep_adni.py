import numpy as np
import pandas as pd
import glob
import os

project_dir = '/home/javier/Desktop/DeepScore/'
adni_dir = '/home/javier/adni/'

df = pd.read_csv(adni_dir + 'moca.csv')
subjects = np.array([os.path.basename(i) for i in sorted(glob.glob(adni_dir+'subjects/*'))])

#take subset of columns
columns = ['PTID', 'VISDATE', 'MOCA']
df = df[columns]

df_new = pd.DataFrame()

for subject in subjects:
    # get label from labels.txt file


    id1 = subject.split('S')[1]
    id2 = subject.split('S')[2]
    original_id = id1+'_S_'+id2
    # find in dataframe
    entries = df[df['PTID'] == original_id]
    # get moca scores and remove NaNs
    entries = entries.dropna(subset=['MOCA'])
    # go trough each subfolder and glob files
    files = sorted(glob.glob(adni_dir+'subjects/'+subject+'/*'))
    for file in files:
        date = os.path.basename(file).split('D')[-1]
        date = date.split('_')[0]
        original_date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        # find closest date
        date_diffs = np.abs((pd.to_datetime(entries['VISDATE']) - pd.to_datetime(original_date)).dt.days)
        closest_idx = date_diffs.idxmin()
        closest_date = entries.loc[closest_idx, 'VISDATE']
        moca_score = entries.loc[closest_idx, 'MOCA']

        # Define your substring to search for
        search_term = os.path.basename(file)
        # Open the file and read line by line
        with open(adni_dir+'labels.txt', 'r') as file:
            for line_number, line in enumerate(file, 1):
                if search_term in line:
                    label = line.strip().split('/')[0]
                    if label == 'mci':
                        label = 'amci'
                    if label == 'ad':
                        label = 'dat'
        
        # subject stats
        # print(f"Subject: {subject}, File Date: {date}, Closest VISDATE: {closest_date}, MOCA Score: {moca_score}, Label: {label}")

        # add to new dataframe
        new_entry = pd.DataFrame({'filename':[search_term],'subject': [subject], 'mri_date': [original_date], 'exam_date': [closest_date], 'label': [label], 'mocatots': [moca_score],})
        df_new = pd.concat([df_new, new_entry], ignore_index=True)

# normative z score for moca
df_subset = df_new[df_new['label'] == 'cn']
mean = df_subset['mocatots'].mean()
std = df_subset['mocatots'].std()
df_new['mocaz'] = (df_new['mocatots'] - mean) / std
# save to csv
df_new.to_csv(project_dir + 'adni.csv', index=False)

