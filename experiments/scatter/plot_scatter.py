from pathlib import Path
import numpy as np
import torch
import os
import pandas as pd
# from DeepScore.utils.dataloaders import get_dataloader
from NeuroMamba.utils.fmri import madc_import, power_import, score_import, mapScores
from scipy import stats
import time
import matplotlib.pyplot as plt
import seaborn as sns
#from DeepScore.utils.trainpost import extract_features
from NeuroMamba.models.NeuroMamba import NeuroMamba
from sklearn.model_selection import LeaveOneOut
from NeuroMamba.utils.dataloaders import get_files, RSFMRI_DATALOADER
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr

mode = "home"
if mode == "home":
    project_dir = "/home/javier/Desktop/DeepScore/"
    madc_file = "/home/javier/Desktop/DeepScore/madc_complete.csv"
    power_file = "/home/javier/Desktop/DeepScore/power_atlas.csv"
    path="/home/javier/madc/"
    score_file = "/home/javier/Desktop/DeepScore/scores.csv"
else:
    project_dir = "/home/javiersc/DeepScore/"
    madc_file = "/home/javiersc/DeepScore/madc_complete.csv"
    power_file = "/home/javiersc/DeepScore/power_atlas.csv"
    path="/home/javiersc/madc/"
    score_file = "/home/javiersc/DeepScore/scores.csv"

power = power_import(power_file)
df = madc_import(madc_file)
coords = power[['X', 'Y', 'Z']].values
score_db = score_import(filename=score_file)

files = np.array(get_files(path, df, type="rest", subject_class="remove_unknown"))
loo = LeaveOneOut()
fold = 1
totalScores = pd.DataFrame() #pd.DataFrame(columns=['subjectID', 'label', 'trueMoCA', 'trueMemory', 'trueLanguage', 'predMoCA', 'predMemory', 'predLanguage'])

for train_index, test_index in tqdm(loo.split(files), total=loo.get_n_splits(files), desc="EVALUATE"):
    #train_files = files[train_index]
    test_file = files[test_index]
    # create datasets
    #train_dataset = RSFMRI_DATALOADER(train_files, transforms=None, database=df, score_database=score_db)
    test_dataset = RSFMRI_DATALOADER(test_file, transforms=None, database=df, score_database=score_db)
    # create dataloaders
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=8)
    # evaluate model
    model = NeuroMamba(n_layers=12, state_dim=32, num_variables=272, score_amount=3).to("cuda")
    model.load_state_dict(torch.load(os.path.join(f'/home/javier/Desktop/DeepScore/weights/loo/neuromamba/b1_{fold}_50.pth')))
    model.eval()

    trueScores = 0.
    predictedScores = 0.
    for idx, (data,info) in enumerate(test_loader):
        trueScores = mapScores(info, num=3, mode="mocaz").to("cuda", dtype=torch.float32)
        inputs = data.to("cuda", dtype=torch.float32)
        with torch.no_grad():
            predictedScores, _ = model(inputs)
    trueScores = trueScores.cpu().numpy().flatten()
    predictedScores = predictedScores.cpu().numpy().flatten()

    subjectIDs = info['subjectID']
    labels = info['label']
    totalScores = pd.concat([totalScores, pd.DataFrame({'subjectID': subjectIDs, 'label': labels, 'True MoCA': trueScores[0], 'True Memory': trueScores[1], 'True Language': trueScores[2], 'Predicted MoCA': predictedScores[0], 'Predicted Memory': predictedScores[1], 'Predicted Language': predictedScores[2]}, index=[0])], ignore_index=True)

    fold += 1

print("R values for MoCA:")
r_moca, pval_moca = pearsonr(totalScores['True MoCA'], totalScores['Predicted MoCA'])
print(f'R = {r_moca:.2f}, p = {pval_moca:.3f}')
print("R values for Memory:")
r_memory, pval_memory = pearsonr(totalScores['True Memory'], totalScores['Predicted Memory'])
print(f'R = {r_memory:.2f}, p = {pval_memory:.3f}')
print("R values for Language:")
r_language, pval_language = pearsonr(totalScores['True Language'], totalScores['Predicted Language'])
print(f'R = {r_language:.2f}, p = {pval_language:.3f}')


pleasant_palette = {
    'cn': '#66c2a5',   # turquoise
    'amci': '#8da0cb', # soft blue
    'dat': '#e78ac3'   # orchid
}
sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc={'figure.figsize':(20,4)})

# MOCA PLOT

g = sns.lmplot(
    data=totalScores, 
    x="True MoCA", y="Predicted MoCA", hue="label", col_order=['cn', 'amci', 'dat'],
    palette=pleasant_palette, facet_kws={'legend_out': False, 'sharex': False, 'sharey': False},
    col="label", height=4, aspect=1, markers=["o", "s", "D"], scatter_kws={"s": 10}, order=1, ci=95
    )

# Loop over axes and add Pearson r
i = 0
for ax, label in zip(g.axes.flat, ['cn', 'amci', 'dat']):
    group = totalScores[totalScores['label'] == label]
    r, pval = pearsonr(group['True MoCA'], group['Predicted MoCA'], alternative='greater')
    ax.annotate(f'R = {r:.2f}, p = {pval:.3f}', 
                xy=(0.01, 0.94), 
                xycoords='axes fraction', 
                fontsize=12, 
                color='black')
    ax.set_xlabel('True MoCA (zscore)')
    if i == 0:
        ax.set_ylabel('Predicted MoCA (zscore)')
    i += 1

plt.savefig(project_dir+"figures/scatter_moca.pdf", dpi=600, bbox_inches='tight')

# MEMORY PLOT
plt.clf()
sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc={'figure.figsize':(20,4)})

g = sns.lmplot(
    data=totalScores, 
    x="True Memory", y="Predicted Memory", hue="label", col_order=['cn', 'amci', 'dat'],
    palette=pleasant_palette, facet_kws={'legend_out': False, 'sharex': False, 'sharey': False},
    col="label", height=4, aspect=1, markers=["o", "s", "D"], scatter_kws={"s": 10}, order=1, ci=95
    )

# Loop over axes and add Pearson r
i = 0
for ax, label in zip(g.axes.flat, ['cn', 'amci', 'dat']):
    group = totalScores[totalScores['label'] == label]
    r, pval = pearsonr(group['True Memory'], group['Predicted Memory'], alternative='greater')
    ax.annotate(f'R = {r:.2f}, p = {pval:.3f}', 
                xy=(0.01, 0.94), 
                xycoords='axes fraction', 
                fontsize=12, 
                color='black')
    ax.set_xlabel('True Memory (zscore)')
    if i == 0:
        ax.set_ylabel('Predicted Memory (zscore)')
    i += 1

plt.savefig(project_dir+"figures/scatter_memory.pdf", dpi=600, bbox_inches='tight')

# LANGUAGE PLOT
plt.clf()
sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc={'figure.figsize':(20,4)})

g = sns.lmplot(
    data=totalScores, 
    x="True Language", y="Predicted Language", hue="label", col_order=['cn', 'amci', 'dat'],
    palette=pleasant_palette, facet_kws={'legend_out': False, 'sharex': False, 'sharey': False},
    col="label", height=4, aspect=1, markers=["o", "s", "D"], scatter_kws={"s": 10}, order=1, ci=95
    )

# Loop over axes and add Pearson r
i = 0
for ax, label in zip(g.axes.flat, ['cn', 'amci', 'dat']):
    group = totalScores[totalScores['label'] == label]
    r, pval = pearsonr(group['True Language'], group['Predicted Language'], alternative='greater')
    ax.annotate(f'R = {r:.2f}, p = {pval:.3f}', 
                xy=(0.01, 0.94), 
                xycoords='axes fraction', 
                fontsize=12, 
                color='black')
    ax.set_xlabel('True Language (zscore)')
    if i == 0:
        ax.set_ylabel('Predicted Language (zscore)')
    i += 1

plt.savefig(project_dir+"figures/scatter_language.pdf", dpi=600, bbox_inches='tight')
      