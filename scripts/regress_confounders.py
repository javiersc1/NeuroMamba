import numpy as np
import pandas as pd

# load scores.csv
scores = pd.read_csv('/home/javier/Desktop/NeuroMamba/scores.csv')
# filter label_alternative == unknown entries
scores = scores[scores['label_alternative'] != 'unknown']
# get mocaz vector
moca = scores['cognition'].values
memory = scores['composite_memory'].values
language = scores['composite_language'].values
#get age vector
age = scores['age'].values
# get education vector
education = scores['education'].values

# regress out age and education from moca, memory, language
from sklearn.linear_model import LinearRegression
def regress_out_confounders(y, confounders):
    model = LinearRegression()
    model.fit(confounders, y)
    y_pred = model.predict(confounders)
    residuals = y - y_pred
    return residuals

confounders = np.column_stack((age, education))
moca_regressed = regress_out_confounders(moca, confounders)
memory_regressed = regress_out_confounders(memory, confounders)
language_regressed = regress_out_confounders(language, confounders)
# add regressed scores to scores dataframe
scores['cognition_regressed'] = moca_regressed
scores['composite_memory_regressed'] = memory_regressed
scores['composite_language_regressed'] = language_regressed

# save scores dataframe to new csv
scores.to_csv('/home/javier/Desktop/NeuroMamba/scores.csv', index=False)

