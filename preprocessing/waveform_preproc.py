#!/usr/bin/env python
# coding: utf-8

# # Waveform Preprocessing

# In[ ]:


# install requirements
import sys
get_ipython().system('{sys.executable} -m pip install --upgrade pip')
get_ipython().system('{sys.executable} -m pip install -r ../requirements.txt')


# ### Load packages!

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

########################################################
# python
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# import warnings
# from time import time
# from copy import copy
# from collections import OrderedDict
# from natsort import natsorted
# import json
# import pickle

########################################################
# sklearn
# warnings.filterwarnings('ignore', message='sklearn.externals.joblib is deprecated in 0.21 and will be removed in 0.23')
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_curve, auc, roc_auc_score

########################################################
# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

########################################################
# waveform package
# https://github.com/-LCP/wfdb-python
import wfdb

########################################################
# set global rnd_seed for reproducibility
rnd_seed = 42
np.random.seed(rnd_seed)

output = './output'

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)


# In[2]:


from plotting import * # load plotting code


# # Load the data

# In[3]:


data_path = '../data/ptb-diagnostic-ecg-database-1.0.0'

target_channel_names = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
n_channels = len(target_channel_names)

time_period = 5 # seconds
sampling_rate = 1000

n_slices_max = 5

# n_rows_list = []
n_rows_slice = sampling_rate*time_period

# possible_Dx = set()


# In[4]:


Dx_classes = [
'BUNDLE BRANCH BLOCK',
'DYSRHYTHMIA',
'MYOCARDIAL INFARCTION',
# 'HEART FAILURE', = # 'CARDIOMYOPATHY', 'HEART FAILURE (NYHA 2)', 'HEART FAILURE (NYHA 3)', 'HEART FAILURE (NYHA 4)',
]


# In[5]:


with open(f'{data_path}/RECORDS', 'r') as f_records:
    records = [x.replace('\n', '') for x in f_records.readlines()]


# In[6]:


with open(f'{data_path}/CONTROLS', 'r') as f_controls:
    controls = [x.replace('\n', '') for x in f_controls.readlines()]


# In[7]:


n_wf_counter_dict = defaultdict(int)
for record in tqdm(records):

    channels, fields = wfdb.rdsamp(f'{data_path}/{record}', channels=list(range(n_channels)))

    record_type = None
    if record in controls:
        record_type = 'CONTROL'
    else:
        Dx_field = 'Reason for admission'
        Dx = [x for x in fields['comments'] if Dx_field in x]
        Dx = Dx[0].replace(f'{Dx_field}: ', '').upper()
        # possible_Dx.add(Dx)

        if Dx in Dx_classes:
            record_type = Dx
        elif Dx == 'CARDIOMYOPATHY' or 'HEART FAILURE' in Dx:
            record_type = 'HEART FAILURE'
        else:
            record_type = 'OTHER'

    channel_names = fields['sig_name']
    if len(set(set(target_channel_names) - set(channel_names))) > 0:
        raise ValueError('Missing some target channels!')

    df_channels = pd.DataFrame(channels, columns=channel_names)
    df_channels = df_channels[target_channel_names]
    n_rows = len(df_channels.index)

    if n_rows < n_rows_slice:
        raise ValueError(f'Only has {n_rows} < n_rows_slice = {n_rows_slice}!')

    n_slices = int(n_rows / n_rows_slice)
    n_slices = min(n_slices, n_slices_max)
    starts = np.random.random(n_slices)

    slice_prop = 1. / float(n_slices)
    starts = (1 - slice_prop)*starts

    for islice in range(n_slices):
        i_slice_start = int(starts[islice]*n_rows)

        i_start = i_slice_start
        i_stop = i_slice_start + n_rows_slice

        plot_waveform(df_channels.iloc[i_start:i_stop], target_channel_names,
                      m_path=f"{output}/{record_type.replace(' ', '_')}", target_period=time_period,
                      fname=f'wf_{n_wf_counter_dict[record_type]}-{islice}', tag='',
                      inline=False)

    n_wf_counter_dict[record_type] += 1


# In[ ]:


# min(n_rows_list)


# In[ ]:


# print(sorted(list(possible_Dx)))


# # Dev

# In[ ]:


# raise ValueError('Stop Here, in Dev!')


# In[ ]:


from plotting import *


# In[ ]:


fields['comments']


# In[ ]:




