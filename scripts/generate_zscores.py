import numpy as np
import pandas as pd
from NeuroMamba.utils.fmri import madc_import, score_import
from NeuroMamba.utils.dataloaders import get_dataframe_entry

dir = '/home/javier/Desktop/DeepScore/'
df_madc = madc_import(dir+"madc_complete.csv")
df_scores = score_import(dir+"scores.csv")

columns = ['mocatots','udsbentd','craftdvr','craftdre','craftvrs','animals_c2','veg_c2','minttots']
for col in columns:
    df_subset = df_scores[df_scores['label_alternative'] == 'cn']
    mean = df_subset[col].mean()
    std = df_subset[col].std()
    print(f'Column: {col}, Mean: {mean}, Std: {std}')
    newcol = col + 'z'
    df_scores[newcol] = (df_scores[col] - mean) / std

df_scores['composite_memory'] = (df_scores['udsbentdz'] + df_scores['craftdvrz'] + df_scores['craftdrez'])*(1/3)
df_scores['composite_language'] = (df_scores['minttotsz'] + df_scores['animals_c2z'] + df_scores['veg_c2z'])*(1/3)
df_scores['cognition'] = df_scores['mocatotsz']

df_scores.to_csv(dir+'scores.csv', index=False)