# load madc_complete.csv
import pandas as pd
import numpy as np

df = pd.read_csv('/home/javier/Desktop/NeuroMamba/madc_complete.csv')
# load scores.csv
scores = pd.read_csv('/home/javier/Desktop/NeuroMamba/scores.csv')

# iterate through each row in scores
for index, row in scores.iterrows():
    subject_id = row['subjectID']
    # split based on e character
    ptid = subject_id.split('e')[0]
    ptid = ptid.split('umm')[1] # UM00001182
    ptid = 'UM000' + ptid
    exam_session = subject_id.split('e')[1]
    # remove leading zeros
    exam_session = exam_session.lstrip('0')
    #print(f"Processing subject {ptid} with exam session {exam_session}")
    # find matching row in df for ptid only
    match = df[(df['ptid'] == ptid)]
    match = match[match['exam_session'] == exam_session]
    # drop rows where mocatots is NaN
    match = match.dropna(subset=['mocatots'])
    # in match, find best entry given form date closest to mri date columns
    match['form_date'] = pd.to_datetime(match['form_date'], errors='coerce', format='%m/%d/%Y')
    match['mri_date'] = pd.to_datetime(match['mri_date'], errors='coerce', format='%m/%d/%Y')
    match['date_diff'] = (match['form_date'] - match['mri_date']).abs()
    match = match.sort_values(by='date_diff')
    # keep only the first row (smallest date_diff)
    match = match.head(1)
    # get age and education vectors
    age = match['age'].values[0]
    education = match['cc_ed_level'].values[0]
    # print age and education for each subject
    print(f"Subject {ptid} with exam session {exam_session} has age: {age} and education: {education}")
    # add age and education to scores dataframe
    scores.at[index, 'age'] = age
    scores.at[index, 'education'] = education
    
# save scores dataframe to new csv
scores.to_csv('/home/javier/Desktop/NeuroMamba/scores2.csv', index=False)

    