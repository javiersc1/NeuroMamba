import os
import numpy as np
import glob
import pandas as pd
import h5py

data = '/storage/madc/conn_project_rsfMRI-Javier-MNI/conn_project_rsfMRI-Javier-MNI/data/'
table = '/home/javiersc/Desktop/scripts/conn_rs_id.csv'
madc = '/home/javiersc/Desktop/madc/'
df = pd.read_csv(table)


for i in range(328):
    print(i)
    connid = str(i+1).zfill(3)
    file = 'ROI_Subject'+connid+'_Session001.mat'
    f = h5py.File(data+file)
    matrix = np.zeros((570, 272))
    for k in range(275):
        if k >= 3:
            array = np.array(f[f['data'][k,0]][:])[0,:]
            matrix[:,k-3] = array
    subject = df.at[i, 'SubjectID']
    os.makedirs(madc+subject, exist_ok=True)
    np.save(madc+subject+'/restingstate'+'.npy', matrix)
