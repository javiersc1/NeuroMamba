import numpy as np
from sklearn.decomposition import FastICA
from NeuroMamba.utils.fmri import mapScores
from NeuroMamba.utils.dataloaders import get_files, RSFMRI_DATALOADER, get_dataframe_entry
from NeuroMamba.utils.fmri import madc_import, score_import, mapScores, mapClasses
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import LeaveOneOut
from NeuroMamba.utils.adni import adni_collate

def GICA(loader, n_components=30, score_amount=3, mode="mocaz"):
    Subjects = []
    Scores = []
    # data
    for idx, (data,info) in enumerate(loader):
        data = data[0,:,:].to(device="cpu").numpy()
        Subjects.append(data)
        scores = mapScores(info, num=score_amount, mode=mode)
        Scores.append(scores)
    # Convert to numpy arrays
    Subjects = np.array(Subjects)
    Scores = np.array(Scores).squeeze()
    num_sub, num_time, num_spatial = Subjects.shape
    Subjects = np.reshape(Subjects, (num_sub*num_time, num_spatial))
    # ---- 4. Run ICA ----
    ica = FastICA(n_components=n_components, random_state=42)
    ica_components = ica.fit_transform(Subjects) 
    #spatial_maps = ica.mixing_.T
    ica_features = ica_components
    # ---- 5. Back-Reconstruction: Split ICA component timecourses per subject ----
    subjectFeatures = []
    start = 0
    for subj in range(num_sub):
        end = start + num_time
        subj_tc = ica_features[start:end, :]  # shape: (n_timepoints, n_components)
        subjectFeatures.append(subj_tc)
        start = end
    subjectFeatures = np.array(subjectFeatures)
    # ---- 6. Extract Features for Each Subject ----
    # Mean absolute value of each component's timecourse per subject
    subjectFeatures = np.mean(np.abs(subjectFeatures), axis=1)
    #subjectFeatures = np.reshape(subjectFeatures, (num_sub, num_time * n_components))

    return subjectFeatures, Scores

def GICA_ADNI(loader, n_components=30, score_amount=3, mode="mocaz", file_mode="train"):
    Subjects = []
    Scores = []
    # data
    for idx, (data,info) in enumerate(loader):
        data, info = adni_collate(data, info, file_mode=file_mode)
        for m in range(data.shape[0]):
            data_m = data[m,:,:].to(device="cpu").numpy()
            Subjects.append(data_m)
            scores = mapScores(info, num=score_amount, mode=mode)
            Scores.append(scores[m])
    # Convert to numpy arrays
    Subjects = np.array(Subjects)
    Scores = np.array(Scores).squeeze()
    num_sub, num_time, num_spatial = Subjects.shape
    #print("Num subs:", num_sub, "Num time:", num_time, "Num spatial:", num_spatial)
    Subjects = np.reshape(Subjects, (num_sub*num_time, num_spatial))
    # ---- 4. Run ICA ----
    ica = FastICA(n_components=n_components, random_state=42)
    ica_components = ica.fit_transform(Subjects) 
    #spatial_maps = ica.mixing_.T
    ica_features = ica_components
    # ---- 5. Back-Reconstruction: Split ICA component timecourses per subject ----
    subjectFeatures = []
    start = 0
    for subj in range(num_sub):
        end = start + num_time
        subj_tc = ica_features[start:end, :]  # shape: (n_timepoints, n_components)
        subjectFeatures.append(subj_tc)
        start = end
    subjectFeatures = np.array(subjectFeatures)
    # ---- 6. Extract Features for Each Subject ----
    # Mean absolute value of each component's timecourse per subject
    subjectFeatures = np.mean(np.abs(subjectFeatures), axis=1)
    #subjectFeatures = np.reshape(subjectFeatures, (num_sub, num_time * n_components))

    return subjectFeatures, np.atleast_1d(Scores)

if __name__ == "__main__":
    mode = "home"
    if mode == "home":
        experiment_dir = "/home/javier/Desktop/DeepScore/experiments/leaveoneout/"
        path = "/home/javier/madc/"
        madc_file = "/home/javier/Desktop/DeepScore/madc_complete.csv"
        score_file = "/home/javier/Desktop/DeepScore/scores.csv"
    elif mode == "server":
        experiment_dir = "/home/javiersc/DeepScore/experiments/leaveoneout/"
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

    subjectFeatures, Scores = GICA(loader, n_components=30, score_amount=score_amount, mode=modescore)