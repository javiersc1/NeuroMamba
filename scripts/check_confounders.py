import numpy as np
import pandas as pd

# load scores.csv
scores = pd.read_csv('/home/javier/Desktop/NeuroMamba/scores.csv')
# filter label_alternative == unknown entries
scores = scores[scores['label_alternative'] != 'unknown']
# get mocaz vector
moca = scores['cognition_regressed'].values
memory = scores['composite_memory_regressed'].values
language = scores['composite_language_regressed'].values
#get age vector
age = scores['age'].values
# get education vector
education = scores['education'].values
# form confounders vector
confounders = np.column_stack((age, education))

from sklearn.linear_model import LinearRegression
def model(y, confounders):
    model = LinearRegression()
    model.fit(confounders, y)
    y_pred = model.predict(confounders)
    return y_pred

from scipy import stats
predictions_moca = model(moca, confounders)
r_moca, p_moca = stats.pearsonr(moca, predictions_moca)

predictions_memory = model(memory, confounders)
r_memory, p_memory = stats.pearsonr(memory, predictions_memory)

predictions_language = model(language, confounders)
r_language, p_language = stats.pearsonr(language, predictions_language)

print(f"MoCA: R={r_moca}, p={p_moca}")
print(f"Memory: R={r_memory}, p={p_memory}")
print(f"Language: R={r_language}, p={p_language}")