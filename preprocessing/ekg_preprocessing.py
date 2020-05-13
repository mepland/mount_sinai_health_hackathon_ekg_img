#!/usr/bin/env python
# coding: utf-8

########################################################
import os
import argparse
import warnings
warnings.filterwarnings('error')
from tqdm import tqdm
from functools import partial

import multiprocessing as mp

import pandas as pd
import numpy as np

from scipy.io import loadmat

from plotting import * # load plotting code

########################################################
# function to process a list of ekgs, in parallel
def process_tranche(in_path, out_path, im_res, slice_time_range, n_slices_max, sampling_freq, channel_names, tranche):
	n_channels = len(channel_names)
	n_samples_per_slice = int(np.ceil(sampling_freq*slice_time_range))
	n_wf_counter_dict = {}

	tranche_name = tranche['name']

	for fname in tranche['fnames']:
		fname = f'{in_path}/{fname}'
		try:
			x = loadmat(f'{fname}.mat')['val']

			with open(f'{fname}.hea', 'r') as f:
				header_data = f.readlines()

			# get gain and offset per channel
			gain_channel = np.zeros(n_channels)
			offset_channel = np.zeros(n_channels)
			for i in range(n_channels):
				h_line_tmp = header_data[i+1].split(' ')
				gain_channel[i] = float(h_line_tmp[2].split('/')[0]) # really an int, but cast to float for safer division later
				offset_channel[i] = int(h_line_tmp[4])

			# get Dx
			Dx = header_data[15].replace('#Dx: ', '')

			# get channel y values
			ch_values = {}
			for ich,ch in enumerate(channel_names):
				ch_values[ch] = (x[ich]-offset_channel[ich])/gain_channel[ich]

			dfp_channels = pd.DataFrame(ch_values)
			n_samples = len(dfp_channels.index)

			# decide n_slices to take, and where to start them
			n_slices = min(int(n_samples / n_samples_per_slice), n_slices_max)
			starts = (1. - (1. / float(n_slices)) )*np.random.random(n_slices)

			m_path = f'{out_path}/{Dx}'
			if Dx not in n_wf_counter_dict.keys():
				n_wf_counter_dict[Dx] = 0
				os.makedirs(m_path, exist_ok=True)

			# print each slice
			for islice in range(n_slices):
				i_slice_start = int(np.ceil(starts[islice]*n_samples))
				i_slice_stop = i_slice_start+n_samples_per_slice

				plot_waveform(dfp_channels.iloc[i_slice_start:i_slice_stop],
					channel_names, sampling_freq,
					m_path=m_path,
					fname=f'wf_{tranche_name}-{n_wf_counter_dict[Dx]}-{islice}',
					tag='', inline=False,
					target_time_range=slice_time_range, target_im_res=im_res,
					run_parallel=True, # Turn off some error checking to speed things up
				)

				n_wf_counter_dict[Dx] += 1

		except (Exception, Warning) as err:
			error_msg = f'Error processing fname = {fname} in tranche_name = {tranche_name}!\n{str(err)}'
			print(error_msg)
			with open('./preprocessing.log', 'a') as f:
				f.write(f'{error_msg}\n')

########################################################
########################################################
if __name__ == '__main__':

	########################################################
	# setup args
	class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter):
		pass

	parser = argparse.ArgumentParser(description='Author: Matthew Epland', formatter_class=lambda prog: CustomFormatter(prog, max_help_position=30))

	parser.add_argument('-i', '--input_path', dest='input_path', type=str, default='../data/Training_Set_2', help='Path to top level directory containing the PhysioNet data.')
	parser.add_argument('-o', '--output_path', dest='output_path', type=str, default='./output', help='Path to output directory. Will actually save to a subdirectory named im_res_{im_res}.')
	parser.add_argument('-n', '--n_ekg_to_process', dest='n_ekg_to_process', type=int, default=-1, help='Number of input EKGs to process, -1 is all.')
	parser.add_argument('-s', '--size', dest='im_res', type=int, default=800, help='Size of output image (800 produces a 800x800 image).')
	parser.add_argument('--slice_time_range', dest='slice_time_range', type=float, default=5., help='Length of time to sample from an EKG (seconds).')
	parser.add_argument('--n_slices_max', dest='n_slices_max', type=int, default=5, help='Maximum number of slices to take from a single EKG.')
	parser.add_argument('--sampling_freq', dest='sampling_freq', type=int, default=500, help='EKG ADC sampling frequency (Hz).')
	parser.add_argument('-j', '--processes', dest='n_processes', type=int, default=1, help='Number of sub-processes run in parallel.')
	parser.add_argument('--n_tranches', dest='n_tranches', type=int, default=-1, help='Number of tranches to create for parallel processing, -1 creates 5*n_processes.')
	parser.add_argument('--seed', dest='rnd_seed', type=int, default=42, help='Random seed for reproducibility.')
	parser.add_argument('-v','--verbose', dest='verbose', action='count', default=0, help='Enable verbose output.')

	# parse the arguments, throw errors if missing any
	args = parser.parse_args()

	# assign to normal variables for convenience
	input_path = args.input_path
	output_path = args.output_path
	n_ekg_to_process = args.n_ekg_to_process
	im_res = args.im_res
	slice_time_range = args.slice_time_range
	n_slices_max = args.n_slices_max
	sampling_freq = args.sampling_freq
	n_processes = args.n_processes
	n_tranches = args.n_tranches
	rnd_seed = args.rnd_seed
	verbose = bool(args.verbose)

	# do some sanity checking
	if im_res < 16 or 2048 < im_res:
		raise ValueError(f'Are you sure you want to run with an output image size of {im_res}? If so, you will have to edit the code...')

	n_cores = mp.cpu_count()
	if n_processes > n_cores:
		raise ValueError(f'Trying to use {n_processes} processes but only have {n_cores} cores!')

	if n_tranches <= 0:
		n_tranches = 5*n_processes

	# setup variables
	channel_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] # TODo hard coded

	output_path = f'{output_path}/im_res_{im_res}' # TODo hard coded

	np.random.seed(rnd_seed)

	# get fnames
	fnames = [f.replace('.mat', '') for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and not f.startswith('.') and f.endswith('mat')]

	# set how many to process
	_n_ekgs = len(fnames)
	if _n_ekgs == 0:
		raise IOError(f'Provided input path {input_path} contained no waveforms')
	if n_ekg_to_process <= 0:
		n_ekg_to_process = _n_ekgs
	if _n_ekgs < n_ekg_to_process:
		print(f'Only found {_n_ekgs} < expected {n_ekg_to_process}, will set n_ekg_to_process = {_n_ekgs} and continue...')
		n_ekg_to_process = _n_ekgs

	fnames = fnames[:n_ekg_to_process]

	# divide up into n_processes tranches
	n_fnames_per_tranche = int(np.ceil(n_ekg_to_process / n_tranches))

	tranches = []
	for itranches in range(n_tranches):
		start = itranches*n_fnames_per_tranche
		stop = start + n_fnames_per_tranche

		if itranches != n_tranches-1:
			tranche_fnames = fnames[start:stop]
		else:
			tranche_fnames = fnames[start:]

		tranches.append({'name': f't{itranches:02d}', 'fnames': tranche_fnames})

	if verbose:
		print(f'n_ekg_to_process = {n_ekg_to_process}, n_tranches = {n_tranches}, n_fnames_per_tranche = {n_fnames_per_tranche}')
		for t in tranches:
			print(f"Tranche {t['name']} contains {len(t['fnames'])} fnames")

	########################################################
	# actually run
	print(f'n_cores = {n_cores}, using n_processes = {n_processes} for n_tranches = {n_tranches}')
	pool = mp.Pool(processes=n_processes)

	process_tranche_partial = partial(process_tranche, input_path, output_path, im_res, slice_time_range, n_slices_max, sampling_freq, channel_names)

	# process_tranche_partial({'name': 'DEBUG', 'fnames': [f for i,f in enumerate(tranches[0]['fnames']) if i < 30]}) # single threaded debugging

	for _ in tqdm(pool.imap_unordered(process_tranche_partial, tranches), total=len(tranches), desc='Tranches'):
		pass

	pool.close()
	pool.join()
