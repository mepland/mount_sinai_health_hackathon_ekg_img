#!/usr/bin/env python
# coding: utf-8

# # Waveform Preprocessing

# In[ ]:


# install requirements
import sys
get_ipython().system('{sys.executable} -m pip install --upgrade pip')
get_ipython().system('{sys.executable} -m pip install -r requirements.txt')


# ### Load packages!

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

########################################################
# python
import pandas as pd
import numpy as np
from natsort import natsorted
from tqdm import tqdm

# import warnings
# from time import time
# from copy import copy
# from collections import OrderedDict
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
# rnd_seed = 42

output = './output'

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)


# In[ ]:


from plotting import * # load plotting code


# # Load the data

# In[4]:


data_path = './data/ptb-diagnostic-ecg-database-1.0.0'

target_channel_names = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
n_channels = len(target_channel_names)

# n_rows_list = []
n_max_rows = 32000

# possible_Dx = set()


# In[5]:


Dx_classes = [
'BUNDLE BRANCH BLOCK',
'DYSRHYTHMIA',
'MYOCARDIAL INFARCTION',
# 'HEART FAILURE', = # 'CARDIOMYOPATHY', 'HEART FAILURE (NYHA 2)', 'HEART FAILURE (NYHA 3)', 'HEART FAILURE (NYHA 4)',
]


# In[6]:


with open(f'{data_path}/RECORDS', 'r') as f_records:
    records = [x.replace('\n', '') for x in f_records.readlines()]


# In[7]:


with open(f'{data_path}/CONTROLS', 'r') as f_controls:
    controls = [x.replace('\n', '') for x in f_controls.readlines()]


# In[8]:


n_wf = 0
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
            continue

    # print(record_type)

    channel_names = fields['sig_name']
    if not set(channel_names).issubset(set(target_channel_names)):
        raise ValueError('Missing some target channels!')

    df_channels = pd.DataFrame(channels, columns=target_channel_names)
    df_channels = df_channels[target_channel_names]

    # print(df_channels.head(4))
    # n_rows_list.append(len(df_channels.index))

    if len(df_channels.index) < n_max_rows:
        raise ValueError(f'Only has {len(df_channels.index)} < n_max_rows = {n_max_rows}!')

    # TODO augment data here by adding noise to y, taking different n_max_rows samples in x
    df_channels_augmented = df_channels.iloc[0:n_max_rows]

    # from plotting import *
    plot_waveform(df_channels_augmented, target_channel_names,
                  m_path=f"{output}/{record_type.replace(' ', '_')}",
                  fname=f'wf_{n_wf}', tag='',
                  inline=False)
    n_wf += 1


# In[ ]:


# min(n_rows_list)


# In[ ]:


# print(sorted(list(possible_Dx)))


# # Dev

# In[ ]:


# raise ValueError('Stop Here, in Dev!')


# In[ ]:


from plotting import *


# In[11]:


fields['comments']


# In[ ]:




