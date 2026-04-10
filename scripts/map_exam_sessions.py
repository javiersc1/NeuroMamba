import numpy as np
import pandas as pd
from datetime import datetime
from NeuroMamba.utils.fmri import madc_import

df = pd.read_csv('/home/javiersc/DeepScore/scripts/exam_sessions.csv', delimiter=' ')
df = df.drop(df.columns[0], axis=1)

madc = madc_import()
madc['exam_session'] = "unknown"

madc_date = madc.copy()
madc_date['mri_date'] = pd.to_datetime(madc_date['mri_date'], format='%d-%b-%Y')

for index, row in df.iterrows():
    subject, date = row.iloc[0], row.iloc[1]
    subject = subject.split('umm')[-1]
    subjectID = subject.split('e')[0]
    examID = subject.split('e')[-1]
    subjectID = "UM000" + subjectID
    date = date.split('_')[0]
    date = datetime.strptime(date, "%Y%m%d")
    #date = date_object.strftime("%d-%b-%Y")
    # subjectID, examID, date are the variables to associate with MADC table
    #madc_filtered = madc[madc['ptid'] == subjectID]
    #pd.to_datetime(madc_filtered['mri_date'], format='%d-%b-%Y')
    #madc_filtered = madc_filtered[madc_filtered['mri_date'] == date]
    # multiple entries since many form dates have the same mri date
    
    #print(madc_date['mri_date'].dt.month)
    #madc.loc[(madc['ptid'] == subjectID) & (madc['mri_date'].dt.month == date.month & madc['mri_date'].dt.year == date.year), 'exam_session'] = examID
    madc.loc[
    (madc['ptid'] == subjectID) &
    (madc_date['mri_date'].dt.month == date.month) &
    (madc_date['mri_date'].dt.year == date.year),
    'exam_session'] = examID

madc.to_csv('/home/javiersc/DeepScore/scripts/madc_complete.csv', index=False)

