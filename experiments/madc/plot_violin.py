import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from NeuroMamba.utils.fmri import *
import numpy as np

mode = "home"
if mode == "home":
    experiment_dir = "/home/javier/Desktop/DeepScore/"
    path = "/home/javier/madc/"
    madc_file = "/home/javier/Desktop/DeepScore/madc_complete.csv"
    score_file = "/home/javier/Desktop/DeepScore/scores.csv"
elif mode == "server":
    experiment_dir = "/home/javiersc/DeepScore/"
    path = "/home/javiersc/madc/"
    madc_file = "/home/javiersc/DeepScore/madc_complete.csv"
    score_file = "/home/javiersc/DeepScore/scores.csv"


df = score_import(score_file)

# remove rows that have label_alternative as 'unknown'
df = df[df['label_alternative'] != 'unknown']


# Define custom colors for groups
pleasant_palette = {
    'cn': '#66c2a5',   # turquoise
    'amci': '#8da0cb', # soft blue
    'dat': '#e78ac3'   # orchid
}

# Set up 1 row and 3 columns of subplots
sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True)
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

# Create one violin plot per axis
sns.violinplot(
    x='label_alternative', y='cognition', data=df, hue='label_alternative',
    palette=pleasant_palette, cut=0,# cut=0 limits violins to the data range
    inner='box', ax=axes[0] # Shows the mean instead of box/point
)
axes[0].set_title("MoCA")
axes[0].set_ylabel('Z-Score Value')
axes[0].set_xlabel('')

sns.violinplot(
    x='label_alternative', y='composite_memory', data=df, hue='label_alternative',
    palette=pleasant_palette, cut=0,# cut=0 limits violins to the data range
    inner='box', ax=axes[1] # Shows the mean instead of box/point
)
axes[1].set_title("Average Memory")
axes[1].set_ylabel('')
axes[1].set_xlabel('')

sns.violinplot(
    x='label_alternative', y='composite_language', data=df, hue='label_alternative',
    palette=pleasant_palette, cut=0,# cut=0 limits violins to the data range
    inner='box', ax=axes[2] # Shows the mean instead of box/point
)
axes[2].set_title("Average Language")
axes[2].set_ylabel('')
axes[2].set_xlabel('')

fig.supxlabel('Disease Category', fontsize=12)
plt.tight_layout()
plt.savefig(experiment_dir+'figures/madc_violin.pdf',dpi=600,bbox_inches='tight')
