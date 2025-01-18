'''
python -W ignore organ_analysis.py
'''

# read AbdomenAtlas3.0.csv into a pandas dataframe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# read the csv file
data = pd.read_csv('AbdomenAtlas3.0.csv')
data.head()

# get the column "pancreas volume (cm^3)"
pancreas_volume = data['pancreas volume (cm^3)']

# get the column "age"
age = data['age']

# get the column "number of pancreatic lesion instances"
num_lesions = data['number of pancreatic lesion instances']

# pair pancreas_volume with age. If any one of them is empty, drop the entrance
volume_age = pd.concat([age, pancreas_volume], axis=1)
volume_age = volume_age.dropna()

# plot the volume-age scatter plot using seaborn, add the regression line, and save the plot as a png file
sns.set(style="whitegrid")
sns.regplot(x='age', y='pancreas volume (cm^3)', data=volume_age)
plt.title('Pancreas Volume vs Age')

# compute the correlation coefficient and add it to the plot, round to 2 decimal places
correlation = volume_age.corr().iloc[0, 1]
plt.text(20, 25, 'r = ' + str(round(correlation, 2)))

plt.savefig('figures/pancreas_volume_age.png')
