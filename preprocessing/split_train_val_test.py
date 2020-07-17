#!/usr/bin/env python
# coding: utf-8

########################################################
import os
import shutil
import argparse
from tqdm import tqdm
from natsort import natsorted
import re
import random


########################################################
########################################################
if __name__ == '__main__':

	########################################################
	# setup args
	class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter):
		pass

	parser = argparse.ArgumentParser(description='Author: Matthew Epland', formatter_class=lambda prog: CustomFormatter(prog, max_help_position=30))


	input_path_default = '../data/preprocessed/im_res_{im_res}/all'
	parser.add_argument('-i', '--input_path', dest='input_path', type=str, default=input_path_default, help='Path to "all" directory containing the class subdirectories.')
	parser.add_argument('-o', '--output_path', dest='output_path', type=str, default=None, help='Path to output directory. Defaults to one level above input.')
	parser.add_argument('-n', '--n_per_class', dest='n_per_class', type=int, default=-2, help='Number of images to take per class. -1 takes the min class number, -2 takes all. Default is -2.')
	parser.add_argument('--test_frac', dest='test_frac', type=float, default=0.15, help='Fraction of files to place in test, default is 0.15.')
	parser.add_argument('--val_frac', dest='val_frac', type=float, default=0.15, help='Fraction of files to place in val, default is 0.15.')
	parser.add_argument('--bywave', dest='by_waveform', action='count', default=1, help='Divide files between train, val, test sets according to the parent waveform, rather than each individual sample.')
	parser.add_argument('--singlesampletest', dest='single_sample_test', action='count', default=0, help='Only return one sample image at random per waveform in the test set.')
	parser.add_argument('-s', '--size', dest='im_res', type=int, default=800, help='Size of image, only used to easily modify the default input_path.')
	parser.add_argument('--extension', dest='ext', type=str, default='png', help='File extension to process, others are ignored. Default is png.')
	parser.add_argument('--seed', dest='rnd_seed', type=int, default=44, help='Random seed for reproducibility.')
	parser.add_argument('-v','--verbose', dest='verbose', action='count', default=0, help='Enable verbose output.')
	# parser.add_argument('--debug', dest='debug', action='count', default=0, help='Enable debugging.')

	# parse the arguments, throw errors if missing any
	args = parser.parse_args()

	# assign to normal variables for convenience
	input_path = args.input_path
	output_path = args.output_path
	n_per_class = args.n_per_class
	test_frac = args.test_frac
	val_frac = args.val_frac
	by_waveform = bool(args.by_waveform)
	single_sample_test = bool(args.single_sample_test)
	im_res = args.im_res
	ext = args.ext
	rnd_seed = args.rnd_seed
	verbose = bool(args.verbose)
	# debug = bool(args.debug)

	# setup vars
	if input_path == input_path_default:
		input_path = input_path_default.format(im_res=im_res)

	if output_path is None:
		output_path = f'{input_path}/..'

	input_path = os.path.abspath(input_path)
	output_path = os.path.abspath(output_path)

	random.seed(rnd_seed)

	folds = {
		'test': test_frac,
		'val': val_frac,
		'train': 1.-val_frac-test_frac,
	}

	# check for existing output dirs
	dirs = [d for d in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, d))]
	abort = [d for d in dirs if d in folds.keys()]
	if len(abort) > 0:
		raise ValueError(f"Can not run while there is existing output directories! Found {output_path}/ {','.join(abort)}")

	# find the class dirs
	classes = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]

	# find files in each class
	class_files = {}
	for c in classes:
		class_path = os.path.join(input_path, c)
		fnames = [f for f in os.listdir(class_path) if os.path.isfile(f'{class_path}/{f}') and f.endswith(f'.{ext}')]
		class_files[c] = {'class_path': class_path,'fnames': natsorted(fnames)}

	if by_waveform:
		class_waveforms = {}
		for c,v in class_files.items():
			waveforms = [re.sub(f'-s\d{2}_PIL\.{ext}', '', f) for f in v['fnames']]
			class_waveforms[c] = {'class_path': v['class_path'],'waveforms': natsorted(list(set(waveforms)))}

	# set min n_per_class from the smallest class, if n_per_class is -1
	if n_per_class == -1:
		l_min = None
		c_min = None
		for c in classes:
			if not by_waveform:
				l = len(class_files[c]['fnames'])
			else:
				l = len(class_waveforms[c]['waveforms'])

			if l_min is None or l < l_min:
				l_min = l
				c_min = c

		if not by_waveform:
			obj = 'files'
		else:
			obj = 'waveforms'
		print(f'{c_min} has the smallest number of {obj} at {l_min}, using this for n_per_class')
		n_per_class = l_min

	# copy each class over
	for c in tqdm(classes, desc='Classes'):
		if verbose:
			print(f'\nProcessing {c}')

		in_class_path = class_files[c]['class_path']

		if not by_waveform:
			objects = class_files[c]['fnames']
		else:
			objects = class_waveforms[c]['waveforms']

		random.shuffle(objects)

		if n_per_class < 0:
			_n_per_class = len(objects)
		else:
			_n_per_class = n_per_class

		objects = objects[:_n_per_class]

		fold_start = 0
		for fold,frac in folds.items():
			if verbose:
				print(f'On {fold}...')

			fold_stop = fold_start + int(_n_per_class * frac)

			out_class_path = f'{os.path.join(output_path, fold)}/{c}'
			os.makedirs(out_class_path, exist_ok=True)

			for o in objects[fold_start:fold_stop]:
				if not by_waveform:
					shutil.copyfile(f'{in_class_path}/{o}', f'{out_class_path}/{o}')
				else:
					# find matching files for this waveform
					fnames = [f for f in class_files[c]['fnames'] if f.startswith(o)]

					if single_sample_test and fold == 'test':
						random.shuffle(fnames)
						fnames = fnames[0:1]

					# print(f'For {o} found {len(fnames)} files:')
					# print(fnames)

					for f in fnames:
						shutil.copyfile(f'{in_class_path}/{f}', f'{out_class_path}/{f}')

			fold_start = fold_stop

	print('Done!')
