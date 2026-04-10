import numpy as np
from sklearn.decomposition import FastICA
from NeuroMamba.utils.fmri import mapScores
from NeuroMamba.utils.dataloaders import get_files, RSFMRI_DATALOADER, get_dataframe_entry
from NeuroMamba.utils.fmri import madc_import, score_import, mapScores, mapClasses
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import LeaveOneOut
from time import sleep
from NeuroMamba.utils.adni import adni_collate

def ALFF(loader, score_amount=3, mode="mocaz"):
    SubjectFeatures = []
    Scores = []

    for idx, (data,info) in enumerate(loader):
        data = data[0,:,:].to(device="cpu").numpy()
        # feature extraction
        freq_magnitudes = np.abs(np.fft.fft(data, axis=0))**0.5
        freq_scale = np.fft.fftfreq(data.shape[0], d=0.8)
        # get indices
        low_freq_indices = (freq_scale <= 0.09) & (freq_scale >= 0.008)
        alff = freq_magnitudes[low_freq_indices, :]
        alff = np.mean(alff, axis=0)
        SubjectFeatures.append(alff)
        #scores
        scores = mapScores(info, num=score_amount, mode=mode)
        Scores.append(scores)


    Scores = np.array(Scores).squeeze()
    SubjectFeatures = np.array(SubjectFeatures)
    
    return SubjectFeatures, Scores

def ALFF_ADNI(loader, score_amount=3, mode="mocaz", file_mode="train"):
    SubjectFeatures = []
    Scores = []

    for idx, (data,info) in enumerate(loader):
        data, info = adni_collate(data, info, file_mode=file_mode)
        scores = mapScores(info, num=score_amount, mode=mode)
        for m in range(data.shape[0]):
            data_m = data[m,:,:].to(device="cpu").numpy()
            # feature extraction
            freq_magnitudes = np.abs(np.fft.fft(data_m, axis=0))**0.5
            freq_scale = np.fft.fftfreq(data_m.shape[0], d=3.0)
            # get indices
            low_freq_indices = (freq_scale <= 0.09) & (freq_scale >= 0.008)
            alff = freq_magnitudes[low_freq_indices, :]
            alff = np.mean(alff, axis=0)
            SubjectFeatures.append(alff)
            Scores.append(scores[m])


    Scores = np.array(Scores).squeeze()
    SubjectFeatures = np.array(SubjectFeatures)
    
    return SubjectFeatures, np.atleast_1d(Scores)

# def fALFF(loader, score_amount=3, mode="mocaz"):
#     SubjectFeatures = []
#     Scores = []

#     for idx, (data,info) in enumerate(loader):
#         data = data[0,:,:].to(device="cpu").numpy()
#         # feature extraction
#         freq_magnitudes = np.abs(np.fft.fft(data, axis=0))**0.5
#         freq_scale = np.fft.fftfreq(data.shape[0], d=0.8)
#         # get indices
#         low_freq_indices = (freq_scale <= 0.09) & (freq_scale >= 0.008)
#         high_freq_indices = (freq_scale <= 0.30) & (freq_scale >= 0.0)
#         # method
#         alff = freq_magnitudes[low_freq_indices, :]
#         alff = np.sum(alff, axis=0)
#         total_ff = freq_magnitudes[high_freq_indices, :]
#         total_ff = np.sum(total_ff, axis=0)
#         falff = alff / total_ff
#         # append features
#         SubjectFeatures.append(falff)
#         scores = mapScores(info, num=score_amount, mode=mode)
#         Scores.append(scores)

#     Scores = np.array(Scores).squeeze()
#     SubjectFeatures = np.array(SubjectFeatures)
    
#     return SubjectFeatures, Scores

if __name__ == "__main__":
    mode = "home"
    if mode == "home":
        path = "/home/javier/madc/"
        madc_file = "/home/javier/Desktop/DeepScore/madc_complete.csv"
        score_file = "/home/javier/Desktop/DeepScore/scores.csv"
    elif mode == "server":
        path = "/home/javiersc/madc/"
        madc_file = "/home/javiersc/DeepScore/madc_complete.csv"
        score_file = "/home/javiersc/DeepScore/scores.csv"

    workers = 8
    type = "rest"
    subject_class = "remove_unknown"
    score_amount = 3
    modescore = "mocaz"
    df = madc_import(filename=madc_file)
    score_db = score_import(filename=score_file)
    files = np.array(get_files(path, df, type=type, subject_class=subject_class))

    dataset = RSFMRI_DATALOADER(files, transforms=None, database=df, score_database=score_db)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=False, num_workers=workers)

    subjectFeatures, Scores = fALFF(loader, score_amount=score_amount, mode=modescore)
    print("Subject Features shape: ", subjectFeatures.shape)
    print("Scores shape: ", Scores.shape)