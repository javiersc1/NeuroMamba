from pathlib import Path
import numpy as np
import torch
import os
import pandas as pd
from NeuroMamba.utils.fmri import madc_import, power_import, score_import, mapScores
from scipy import stats
import time
import matplotlib.pyplot as plt
import seaborn as sns
from NeuroMamba.models.NeuroMamba import *
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter1d

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
totalInputs = np.load(project_dir + "figures/totalInputs.npy")
totalScores = np.load(project_dir + "figures/totalScores.npy")
totalLabels = np.load(project_dir + "figures/totalLabels.npy")

def colorline(x, y, c, ax, sigma=5, **lc_kwargs):
    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    c = gaussian_filter1d(c, sigma=sigma)
    c = c / c.max()

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    y = np.asarray(y)
    x = np.arange(y.shape[0]) if x is None else np.asarray(x)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    ax.set_xlim(0, x[-1])
    ax.set_ylim(y.min(), y.max())

    return ax.add_collection(lc)

subject = 122

model = NeuroMamba(n_layers =12, state_dim = 32, num_variables=272, score_amount=3).to("cuda")
model.load_state_dict(torch.load(f'/home/javier/Desktop/DeepScore/weights/loo/neuromamba/{subject+1}_100.pth', weights_only=True))
print("subject:", subject)
print("input shape:", totalInputs[subject].shape)
print("score:", totalScores[subject])
print("label:", totalLabels[subject])

input_data = torch.tensor(totalInputs[subject,:,:], dtype=torch.float32).to("cuda").unsqueeze(0)
input_data.requires_grad_()
output,_ = model(input_data)
output[:,0].backward()
saliency = input_data.grad.abs()
saliency_just_time = saliency.sum(dim=2)
print(output[:,0].item())

c = saliency.cpu().numpy().squeeze()

rois = [77, 170, 177, 221, 93]
titles = ['Parahippocampal \n Gyrus', 'Cuneus', 'Inferior Parietal \n Lobule', 'Cingulate \n Gyrus', 'Precuneus']
num = len(rois)
fig, axs = plt.subplots(num, 1, figsize=(20, 2*num))
time = np.arange(570)*0.8
for i, roi in enumerate(rois):
    axs[i].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.25)
    lines = colorline(time, totalInputs[subject,:,roi], c[:,roi], axs[i], sigma=10, linewidth=1, cmap="jet")
    axs[i].get_xaxis().set_visible(False)
    axs[i].set_ylabel(titles[i])
    axs[i].set_yticklabels([])
    axs[i].set_yticks([])
    if i == num - 1:
        axs[i].get_xaxis().set_visible(True)
        axs[i].set_xlabel("Time (s)")
    #fig.colorbar(lines, ax=axs[i])  # add a color legend
fig.colorbar(lines, ax=axs, orientation='vertical', label='', fraction=0.04, pad=0.01, shrink=1.0)
fig.savefig(project_dir+'figures/saliency.pdf', dpi=600, bbox_inches='tight')