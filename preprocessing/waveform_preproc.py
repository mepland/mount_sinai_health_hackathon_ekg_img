#!/usr/bin/env python
# coding: utf-8

# # Waveform Preprocessing
import sys
get_ipython().system('{sys.executable} -m pip install --upgrade pip');
get_ipython().system('{sys.executable} -m pip install -r ../requirements.txt');
# ### Load packages!

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

########################################################
# python
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from scipy.io import loadmat
  
########################################################
# plotting
from plotting import * # load plotting code
get_ipython().run_line_magic('matplotlib', 'inline')

########################################################
# set global rnd_seed for reproducibility
rnd_seed = 42
np.random.seed(rnd_seed)

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)


# ***
# # Setup Variables and Functions

# In[ ]:


# input vars for this sample
data_path = '../data/PhysioNetChallenge2020_Training_CPSC/Training_WFDB'

n_ekgs_total = 6877
sampling_freq = 500 # Hz

channel_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
n_channels = len(channel_names)

Dx_classes = {
'Normal': 'Normal sinus rhythm',
'AF': 'Atrial fibrillation',
'I-AVB': 'Airst-degree atrioventricular block',
'LBBB': 'Left bundle branch block',
'PAC': 'Premature atrial complex',
'PVC': 'Premature ventricular complex',
'RBBB': 'Right bundle branch block',
'STD': 'ST-segment depression',
'STE': 'ST-segment elevation',
}

# output vars
im_res=800
output_path = f'./output/im_res_{im_res}'
slice_time_range = 5 # seconds
n_slices_max = 5 # max number of slice_time_range length slices to take from one original waveform


# In[ ]:


def load_challenge_data(m_path, fname):
    if '.mat' in fname or '.hea' in fname:
        raise ValueError(f'Expected fname = {fname} to NOT have any extensions .mat or .hea')

    mat_fname =  f'{m_path}/{fname}.mat'
    try:
        x = loadmat(mat_fname)['val']
    except EnvironmentError:
        raise EnvironmentError(f'Could not load mat file {mat_fname}')

    header_fname =  f'{m_path}/{fname}.hea'
    try:
        with open(header_fname, 'r') as f:
            header_data_raw = f.readlines()

            # clean any whitespace to single spaces
            header_data = [' '.join(l.split()) for l in header_data_raw]

            # pull metadata from first line
            h_line = header_data[0].split(' ')

            _n_leads = int(h_line[1])
            if _n_leads != n_channels:
                raise ValueError(f'Expected {n_channels} but found {_n_leads}!')

            _sampling_freq = int(h_line[2])
            if _sampling_freq != sampling_freq:
                raise ValueError(f'Header reports a {_sampling_freq} Hz sampling rate, but expected {sampling_freq}!')

            _n_samples = int(h_line[3])
            if _n_samples != x.shape[1]:
                raise ValueError(f'Header reports {_n_samples} samples, but .mat only has {x.shape[1]}!')

            # get gain per lead
            gain_lead = np.zeros(n_channels)
            offset_lead = np.zeros(n_channels)
            for i in range(n_channels):
                h_line_tmp = header_data[i+1].split(' ')
                gain_split = h_line_tmp[2].split('/')
                if gain_split[1] != 'mV':
                    raise ValueError(f'Expected mV units, but found {gain_split[1]}!')
                gain_lead[i] = int(gain_split[0])
                offset_lead[i] = int(h_line_tmp[4])

            # get Dx
            Dx_line = header_data[15]
            if 'Dx' not in Dx_line:
                raise ValueError('Did not find Dx information on line 16!')
            Dx = Dx_line.replace('#Dx: ', '')

    except EnvironmentError:
        raise EnvironmentError(f'Could not load header file {header_fname}')

    results = {}
    for ich,ch in enumerate(channel_names):
        results[ch] = (x[i]-offset_lead[i])/gain_lead[i]
    dfp = pd.DataFrame(results)

    return dfp, Dx


# ***
# # Load the data and plot waveforms

# In[ ]:


input_files = []
for f in os.listdir(data_path):
    if os.path.isfile(os.path.join(data_path, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
        input_files.append(f.replace('.mat', ''))

_n_ekgs = len(input_files)
if _n_ekgs != n_ekgs_total:
    raise ValueError(f'Only found {_n_ekgs} != expected {n_ekgs_total}!')


# In[ ]:


# allowed_Dx_classes = [] # TODo


# In[ ]:


n_wf_counter_dict = defaultdict(int)
n_rows_per_slice = sampling_freq*slice_time_range
for input_file in tqdm(input_files):

    dfp_channels, Dx = load_challenge_data(data_path, input_file)

    record_type = Dx
#    if Dx in allowed_Dx_classes:
#        record_type = Dx
#    elif Dx in ['TODo', 'TODo']:
#        record_type = 'TODo'
#    else:
#        record_type = 'Other'
    record_type = record_type.replace(' ', '_')

    n_rows = len(dfp_channels.index)
    if n_rows < n_rows_per_slice:
        raise ValueError(f'Only has {n_rows} < n_rows_per_slice = {n_rows_per_slice}!')

    n_slices = min(int(n_rows / n_rows_per_slice), n_slices_max)
    slice_prop = 1. / float(n_slices)

    starts = np.random.random(n_slices)
    starts = (1 - slice_prop)*starts

    for islice in range(n_slices):
        i_slice_start = int(starts[islice]*n_rows)

        plot_waveform(dfp_channels.iloc[i_slice_start:i_slice_start+n_rows_per_slice],
                      channel_names, sampling_freq,
                      m_path=f'{output_path}/{record_type}',
                      fname=f'wf_{n_wf_counter_dict[record_type]}-{islice}',
                      tag='', inline=False,
                      target_time_range=slice_time_range, target_im_res=im_res,
                      )

    n_wf_counter_dict[record_type] += 1


# ***
# # Dev

# In[ ]:


raise ValueError('Stop Here, in Dev!')


# In[ ]:


from plotting import *


# In[ ]:


dfp_channels, Dx = load_challenge_data(data_path, fname='A0001', target_time_range=slice_time_range, target_im_res=im_res)


# In[ ]:


Dx


# In[ ]:


dfp_channels


# In[ ]:


plot_waveform(dfp_channels, channel_names, sampling_freq, output_path, inline=False)


# In[ ]:




