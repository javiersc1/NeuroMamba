import numpy as np
import pandas as pd
from NeuroMamba.utils.fmri import madc_import, score_import
from NeuroMamba.utils.dataloaders import get_dataframe_entry

dir = '/home/javier/Desktop/DeepScore/'
df_madc = madc_import(dir+"madc_complete.csv")
df_scores = score_import(dir+"scores.csv")

df_madc['label_alternative'] = 'unknown'

conditions = ['Cognitive impairment due to systemic disease/medical illness (as indicated on Form D2)',
              'Vascular Brain Injury (based on clinical or imaging evidence)', 
              'Cognitive impairment due to other neurologic, genetic, or infectious conditions not listed',
               'FTLD NOS', 'Anxiety disorder', 'Traumatic brain injury', 'Other psychiatric disease',
               'Cogntivie impairment due to medications', 'FTLD with motor neuron disease', 'Corticobasal degeneration (CBD)',
               'Progressive supranuclear palsy (PSP)', 
                ]

#print(df_madc['madc_primary_etiology'].unique())

for index, row in df_madc.iterrows():
    #if row['madc_primary_etiology'] not in conditions:
    if row['label'] != 'unknown':
        if row['madc_dx'] == 'Normal':
            df_madc.at[index, 'label_alternative'] = 'cn'

        if row['madc_dx'] == 'Amnestic MCI-single domain':
            df_madc.at[index, 'label_alternative'] = 'amci'
        if row['madc_dx'] == 'Amnestic MCI-multiple domains':
            df_madc.at[index, 'label_alternative'] = 'amci'

        if row['madc_dx'] == 'Non-amnestic MCI-single domain':
            df_madc.at[index, 'label_alternative'] = 'namci'
        if row['madc_dx'] == 'Non-amnestic MCI-multiple domains':
            df_madc.at[index, 'label_alternative'] = 'namci'

        if row['madc_dx'] == 'Amnestic multidomain dementia syndrome':
            df_madc.at[index, 'label_alternative'] = 'dat'

    # if row['madc_dx'] == 'Non-amnestic multidomain dementia, not PCA, PPA, bvFTD, or DLB syndrome':
    #     df_madc.at[index, 'label_alternative'] = 'dat'
    # if row['madc_dx'] == 'Lewy body dementia syndrome':
    #     df_madc.at[index, 'label_alternative'] = 'dat'

df_madc.to_csv(dir+'madc_complete.csv', index=False)

df_scores['label_alternative'] = 'unknown'

for index, row in df_scores.iterrows():
    session = get_dataframe_entry(df_madc, row['subjectID'])
    df_scores.at[index, 'label_alternative'] = session['label_alternative']

df_scores.to_csv(dir+'scores.csv', index=False)



    