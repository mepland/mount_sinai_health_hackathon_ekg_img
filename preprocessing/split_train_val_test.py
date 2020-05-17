#!/usr/bin/env python
# coding: utf-8

########################################################
import os
import shutil
import argparse
from tqdm import tqdm

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
	parser.add_argument('-n', '--n_per_class', dest='n_per_class', type=int, default=-1, help='Number of images to take per class. -1 takes the min class number, -2 takes all. Default is -1.')
	parser.add_argument('--test_frac', dest='test_frac', type=float, default=0.15, help='Fraction of files to place in test, default is 0.15.')
	parser.add_argument('--val_frac', dest='val_frac', type=float, default=0.15, help='Fraction of files to place in val, default is 0.15.')
	parser.add_argument('-s', '--size', dest='im_res', type=int, default=600, help='Size of image, only used to easily modify the default input_path.')
	parser.add_argument('--extension', dest='ext', type=str, default='png', help='File extension to process, others are ignored. Default is png.')
	parser.add_argument('--seed', dest='rnd_seed', type=int, default=42, help='Random seed for reproducibility.')
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
		class_files[c] = {'class_path': class_path,'fnames': fnames}

	# set min n_per_class from the smallest class, if n_per_class is -1
	if n_per_class == -1:
		l_min = None
		c_min = None
		for c,v in class_files.items():
			l = len(v['fnames'])
			if l_min is None or l < l_min:
				l_min = l
				c_min = c
		print(f'{c_min} has the smallest number of files at {l_min}, using this for n_per_class')
		n_per_class = l_min

	# copy each class over
	for c in tqdm(classes, desc='Classes'):
		if verbose:
			print(f'\nProcessing {c}')

		in_class_path = class_files[c]['class_path']
		fnames = class_files[c]['fnames']

		random.shuffle(fnames)

		if n_per_class < 0:
			n_per_class = len(fnames)

		fnames = fnames[:n_per_class]

		fold_start = 0
		for fold,frac in folds.items():
			fold_stop = fold_start + int(n_per_class * frac)

			out_class_path = f'{os.path.join(output_path, fold)}/{c}'
			os.makedirs(out_class_path, exist_ok=True)

			for f in fnames[fold_start:fold_stop]:
				shutil.copyfile(f'{in_class_path}/{f}', f'{out_class_path}/{f}')

			fold_start = fold_stop

	print('Done!')
