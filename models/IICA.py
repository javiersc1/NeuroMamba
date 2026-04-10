import numpy as np
from sklearn.decomposition import FastICA
from NeuroMamba.utils.fmri import mapScores
from NeuroMamba.utils.dataloaders import get_files, RSFMRI_DATALOADER, get_dataframe_entry
from NeuroMamba.utils.fmri import madc_import, score_import, mapScores, mapClasses
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import LeaveOneOut
from NeuroMamba.utils.adni import adni_collate

def IICA(loader, n_components=30, score_amount=3, mode="mocaz", algorithm='parallel'):
    SubjectFeatures = []
    Scores = []

    for idx, (data,info) in enumerate(loader):
        data = data[0,:,:].to(device="cpu").numpy()
        #print("Data shape: ", data.shape)
        transformer = FastICA(n_components=n_components,random_state=42,max_iter=1000, algorithm=algorithm)
        transformer.fit(data.T)
        components = transformer.components_
        #print("Components shape: ", components.shape)
        features = np.mean(np.abs(components), axis=1)
        SubjectFeatures.append(features)
        scores = mapScores(info, num=score_amount, mode=mode)
        Scores.append(scores)


    Scores = np.array(Scores).squeeze()
    SubjectFeatures = np.array(SubjectFeatures)
    
    return SubjectFeatures, Scores


def IICA_ADNI(loader, n_components=30, score_amount=3, mode="mocaz", file_mode="train"):
    SubjectFeatures = []
    Scores = []

    for idx, (data,info) in enumerate(loader):
        data, info = adni_collate(data, info, file_mode=file_mode)
        for m in range(data.shape[0]):
            data_m = data[m,:,:].to(device="cpu").numpy()
            #print("Data shape: ", data_m.shape)
            transformer = FastICA(n_components=n_components,random_state=42,max_iter=1000, algorithm='deflation')
            transformer.fit(data_m.T)
            components = transformer.components_
            features = np.mean(np.abs(components), axis=1)
            SubjectFeatures.append(features)
            scores = mapScores(info, num=score_amount, mode=mode)
            Scores.append(scores[m])

    Scores = np.array(Scores).squeeze()
    SubjectFeatures = np.stack(SubjectFeatures)

    if len(SubjectFeatures.shape) == 1:
        SubjectFeatures = SubjectFeatures[None, :]
    
    return SubjectFeatures, np.atleast_1d(Scores)

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

    subjectFeatures, Scores = IICA(loader, n_components=30, score_amount=score_amount, mode=modescore)
    print("Subject Features shape: ", subjectFeatures.shape)
    print("Scores shape: ", Scores.shape)