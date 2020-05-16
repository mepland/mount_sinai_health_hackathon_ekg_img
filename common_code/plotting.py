# python
from operator import itemgetter
# import collections
# from itertools import cycle
# import logging
# logging.basicConfig(level=logging.WARNING)
# from ast import literal_eval
# import warnings

########################################################
# plotting
import matplotlib as mpl
mpl.use('Agg', warn=False)
mpl.rcParams['font.family'] = ['HelveticaNeue-Light', 'Helvetica Neue Light', 'Helvetica Neue', 'Helvetica', 'Arial', 'Lucida Grande', 'sans-serif']
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.top']           = True
mpl.rcParams['ytick.right']         = True
mpl.rcParams['xtick.direction']     = 'in'
mpl.rcParams['ytick.direction']     = 'in'
mpl.rcParams['xtick.labelsize']     = 13
mpl.rcParams['ytick.labelsize']     = 13
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['xtick.major.width']   = 0.8  # major tick width in points
mpl.rcParams['xtick.minor.width']   = 0.8  # minor tick width in points
mpl.rcParams['xtick.major.size']    = 7.0  # major tick size in points
mpl.rcParams['xtick.minor.size']    = 4.0  # minor tick size in points
mpl.rcParams['xtick.major.pad']     = 1.5  # distance to major tick label in points
mpl.rcParams['xtick.minor.pad']     = 1.4  # distance to the minor tick label in points
mpl.rcParams['ytick.major.width']   = 0.8  # major tick width in points
mpl.rcParams['ytick.minor.width']   = 0.8  # minor tick width in points
mpl.rcParams['ytick.major.size']    = 7.0  # major tick size in points
mpl.rcParams['ytick.minor.size']    = 4.0  # minor tick size in points
mpl.rcParams['ytick.major.pad']     = 1.5  # distance to major tick label in points
mpl.rcParams['ytick.minor.pad']     = 1.4  # distance to the minor tick label in points

# Set the default color cycle TODO test
# mpl.rcParams['axes.prop_cycle'] = cycle(tableau.Tableau_10.colors)
# mpl.rcParams['axes.prop_cycle'] = 'T10'

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import matplotlib.ticker as ticker

from palettable import tableau # https://jiffyclub.github.io/palettable/

default_colors = tableau.Tableau_20.colors
# default_colors_mpl = tableau.Tableau_20.mpl_colors
# default_colors10_mpl = tableau.Tableau_10.mpl_colors
# default_colors20_mpl = tableau.Tableau_20.mpl_colors

########################################################
# load package wide variables
from .configs import *

########################################################
# Set common plot parameters
vsize = 11 # inches
# aspect ratio width / height
aspect_ratio_single = 4.0/3.0
aspect_ratio_multi = 1.0

plot_png=False
png_dpi=200

std_ann_x = 0.80
std_ann_y = 0.95

std_cmap = cm.plasma
std_cmap_r = cm.plasma_r

########################################################
top_line_std_ann = ''

########################################################
def date_ann(dt_start, dt_stop):
	if not (isinstance(dt_start, dt.datetime) and isinstance(dt_stop, dt.datetime)) or not (isinstance(dt_start, dt.date) and isinstance(dt_stop, dt.date)):
		return ''

	if dt_start.month == 1 and dt_start.day == 1 and dt_stop.month == 12 and dt_stop.day == 31:
		# interval is a whole number of years
		if dt_start.year == dt_stop.year:
			return str(dt_start.year)
		else:
			return f'{dt_start.year} - {dt_stop.year}'
	else:
		return f"{dt_start.strftime('%Y-%m-%d')} - {dt_stop.strftime('%Y-%m-%d')}"

########################################################
def ann_text_std(dt_start, dt_stop, ann_text_std_add, ann_text_hard_coded=None, gen_date=False):
	e1 = ''
	if top_line_std_ann != '':
		e1 = f'{top_line_std_ann}\n'

	date_ann_str = date_ann(dt_start, dt_stop)
	e2 = ''
	if date_ann_str != '':
		e2 = f'{date_ann_str}\n'

	e3 = ''
	if gen_date:
		e3 = f"Generated: {dt.datetime.now().strftime('%Y-%m-%d')}\n"

	e4 = ''
	if ann_text_hard_coded is not None and ann_text_hard_coded != '':
		e4 = f'{ann_text_hard_coded}\n'

	e5 = ''
	if ann_text_std_add is not None and ann_text_std_add != '':
		e5 = f'{ann_text_std_add}'

	ann_str = f'{e1}{e2}{e3}{e4}{e5}'

	return ann_str.strip('\n')

########################################################
def _setup_vars(ann_texts_in, x_axis_params, y_axis_params):
	ann_texts = []
	if ann_texts_in is not None:
		ann_texts = list(ann_texts_in)
	if not isinstance(x_axis_params, dict):
		x_axis_params = dict()
	if not isinstance(y_axis_params, dict):
		y_axis_params = dict()
	return ann_texts, x_axis_params, y_axis_params

########################################################
def set_ax_limits(_ax, x_axis_params, y_axis_params, allow_maxMult=False):
	x_min_auto, x_max_auto = _ax.get_xlim()
	x_min = x_axis_params.get('min', None)
	x_max = x_axis_params.get('max', None)
	if x_min is None:
		x_min = x_min_auto
	if x_max is None:
		x_max = x_max_auto

	y_min_auto, y_max_auto = _ax.get_ylim()
	y_min = y_axis_params.get('min', None)
	y_max = y_axis_params.get('max', None)
	y_maxMult = None
	if allow_maxMult:
		y_maxMult = y_axis_params.get('maxMult', None)
	if y_min is None:
		y_min = y_min_auto

	if y_maxMult is not None:
		y_max = y_min + y_maxMult*(y_max_auto-y_min)
	elif y_max is None:
		y_max = y_max_auto

	_ax.set_xlim(x_min, x_max)
	_ax.set_ylim(y_min, y_max)

########################################################
def clean_ax(_ax, x_axis_params, y_axis_params, turn_off_axes=False):
	if turn_off_axes:
		_ax.axis('off')
	else:
		_ax.set_xlabel(x_axis_params.get('axis_label', ''))
		_ax.set_ylabel(y_axis_params.get('axis_label', ''))
		_ax.xaxis.label.set_size(20)
		_ax.yaxis.label.set_size(20)
		_ax.xaxis.set_tick_params(labelsize=15)
		_ax.yaxis.set_tick_params(labelsize=15)

		if x_axis_params.get('log', False):
			_ax.set_xscale('log')

		if y_axis_params.get('log', False):
			_ax.set_yscale('log')

########################################################
def ann_and_save(_fig, ann_texts, inline, m_path, fname, tag, ann_text_origin_x=std_ann_x, ann_text_origin_y=std_ann_y, forced_text_size=None):
	if ann_texts is not None:
		for text in ann_texts:
			if forced_text_size is not None:
				text_size = forced_text_size
			else:
				text_size = text.get('size', 18)

			plt.figtext(ann_text_origin_x+text.get('x', 0.0), ann_text_origin_y+text.get('y', 0.0), text.get('label', 'MISSING'), ha=text.get('ha', 'left'), va='top', size=text_size, backgroundcolor='white')

	_fig.tight_layout()
	if inline:
		_fig.show()
	else:
		os.makedirs(m_path, exist_ok=True)
		if plot_png:
			_fig.savefig(f'{m_path}/{fname}{tag}.png', dpi=png_dpi)
		_fig.savefig(f'{m_path}/{fname}{tag}.pdf')
		plt.close('all')

########################################################
def unnormalize_im(im, std_unnormalize, mean_unnormalize):
	if std_unnormalize is not None and mean_unnormalize is not None:
		im_unnorm = np.zeros(im.shape)
		for channel in range(im.shape[0]):
			im_unnorm[channel] = std_unnormalize[channel]*im[channel] + mean_unnormalize[channel]
		# now clip to [0,1] to deal with any rounding errors and prevent later warning from imshow
		im_unnorm = np.clip(im_unnorm, 0., 1.)

		return im_unnorm
	else:
		return im

########################################################
def plot_hists(hist_dicts, m_path, fname='hist', tag='', dt_start=None, dt_stop=None, inline=False, ann_text_std_add=None, ann_texts_in=None, binning={'nbins': 10}, x_axis_params=None, y_axis_params=None): # , label_values=False
# Standard
# hist_dict = {'values': , 'weights': None, 'label': None, 'histtype': 'step', 'stacked': False, 'density': False, 'c': None, 'lw': 2}
# Precomputed
# hist_dict = {'hist_data': {'bin_edges': [], 'hist':[]}, ... AND binning = {'use_hist_data': True}
# as bar graph
# hist_dict = {'plot_via_bar': False, 'fill': True, 'ec': None, 'ls': None, 'label_values': False

# x_axis_params={'axis_label':None, 'min':None, 'max':None, 'units': '', 'log': False}, y_axis_params={'axis_label':None, 'min':None, 'max':None, 'maxMult':None, 'log': False, 'show_bin_size': True}

	ann_texts, x_axis_params, y_axis_params = _setup_vars(ann_texts_in, x_axis_params, y_axis_params)
	x_axis_params['axis_label'] = x_axis_params.get('axis_label', 'Bins')

	bin_edges = binning.get('bin_edges', [])
	nbins = binning.get('nbins', None)
	bin_size = binning.get('bin_size', None)
	bin_size_str_fmt = binning.get('bin_size_str_fmt', '.2f')

	for ihd,hd in enumerate(hist_dicts):
		if len(hd.get('values', [])) > 0:
			_values = hd['values']
		elif len(hd.get('hist_data', {}).get('bin_edges', [])) > 0:
			_values = hd['hist_data']['bin_edges']

			if bin_edges == [] and binning.get('use_hist_data', False):
				bin_edges = list(_values)

		else:
			raise ValueError('Should not end up here, re-evaluate your hist_dicts and binning inputs!')

		if ihd == 0:
			_bin_min = min(_values)
			_bin_max = max(_values)
		else:
			_bin_min = min(_bin_min, min(_values))
			_bin_max = max(_bin_max, max(_values))

	_bin_min = binning.get('min', _bin_min)
	_bin_max = binning.get('max', _bin_max)

	if (isinstance(bin_edges, list) or isinstance(bin_edges, np.array)) and len(bin_edges) >= 2:
		# possibly variable size bins from bin_edges
		nbins = len(bin_edges)-1
		bin_edges = np.array(bin_edges)
		if bin_size is not None:
			bin_size_str = f'{bin_size:{bin_size_str_fmt}}'
		else:
			bin_size_str = 'Variable'
	elif bin_size is not None and bin_size > 0.:
		# fixed bin_size
		nbins = int(round((_bin_max-_bin_min)/bin_size))
		bin_edges = np.linspace(_bin_min, _bin_max, nbins+1)
		bin_size = (_bin_max-_bin_min)/nbins
		bin_size_str = f'{bin_size:{bin_size_str_fmt}}'
	elif nbins is not None and nbins > 0:
		# fixed number of bins
		bin_edges = np.linspace(_bin_min, _bin_max, nbins+1)
		bin_size = (_bin_max-_bin_min)/nbins
		bin_size_str = f'{bin_size:{bin_size_str_fmt}}'
	else:
		print(binning)
		raise ValueError('Can not work with this binning dict!')

	leg_objects = []

	fig, ax = plt.subplots(num=fname)
	fig.set_size_inches(aspect_ratio_single*vsize, vsize)

	for hd in hist_dicts:
		if len(hd.get('values', [])) > 0:
			_hist = hd['values']
			_bins=bin_edges
			_weights=hd.get('weights', None)
		elif len(hd.get('hist_data', {}).get('bin_edges', [])) > 0:
			# results are already binned, so fake the input by giving 1 count to the middle of each bin, then multiplying by the appropriate weight
			_bin_edges = hd['hist_data']['bin_edges']
			if list(bin_edges) != list(_bin_edges):
				print('Warning this hist_data dict does not have the same bin edges as the first, not expected! Will try to continue but bins are not going to line up and may be beyond the axis range')
				print(hd)

			_hist = []
			for ibin in range(len(_bin_edges)-1):
				bin_min = _bin_edges[ibin]
				bin_max = _bin_edges[ibin+1]
				_hist.append(bin_min + 0.5*(bin_max - bin_min))

			_bins = _bin_edges
			_weights = hd['hist_data']['hist']

		if not hd.get('plot_via_bar', False):

			_plotted_hist, _plotted_bin_edges, _plotted_patches = ax.hist(_hist, bins=_bins, weights=_weights, label=hd.get('label', None), histtype=hd.get('histtype', 'step'), stacked=hd.get('stacked', False), density=hd.get('density', False), log=y_axis_params.get('log', False), color=hd.get('c', None), linewidth=hd.get('lw', 2))

			_label = _plotted_patches[0].get_label()
			if _label is not None and _label is not '':
				leg_objects.append(_plotted_patches[0])

		else:
			# plot via ax.bar instead of ax.hist - better for some use cases with variable bins
			if len(_bins)-1 != len(_weights):
				raise ValueError('TODO write numpy code to histogram the values')

			_nbins = len(_bins)-1
			x_axis_labels = []
			x_axis_ticks = np.arange(_nbins)
			for i in range(_nbins):
				upper_inequality = '$<$'
				if i == nbins-1:
					upper_inequality = '$\leq$'
				x_axis_labels.append('{low} $\leq$ {var} {upper_inequality} {high}'.format(low=my_large_num_formatter(_bins[i], e_precision=0), high=my_large_num_formatter(_bins[i+1], e_precision=0), var=x_axis_params.get('axis_label', 'Binned Variable'), upper_inequality=upper_inequality))

			hist_bin_values = np.array(_weights)
			if hd.get('density', False):
				hist_bin_values = np.divide(hist_bin_values, float(sum(hist_bin_values)))

			_label=hd.get('label', None)
			ax.bar(x_axis_ticks, hist_bin_values, width=0.5, label=_label, log=y_axis_params.get('log', False), color=hd.get('c', None), linewidth=hd.get('lw', 2), fill=hd.get('fill', True), edgecolor=hd.get('ec', None), ls=hd.get('ls', None))
			if _label is not None:
				handles, labels = ax.get_legend_handles_labels()
				leg_objects.append(handles[labels.index(_label)])

			ax.set_xticklabels(x_axis_labels, rotation=45, ha='right')
			ax.set_xticks(x_axis_ticks)
			ax.tick_params(axis='x', which='both', length=0)

			if hd.get('label_values', False):
				rects = ax.patches
				for rect, label in zip(rects, hist_bin_values):
					height = rect.get_height()
					ax.text(rect.get_x() + rect.get_width() / 2, height, my_large_num_formatter(label, e_precision=0), ha='center', va='bottom')

	y_label = y_axis_params.get('axis_label', '$N$')
	if y_axis_params.get('show_bin_size', True):
		y_label = f'{y_label} / {bin_size_str}'

	x_units = x_axis_params.get('units', '')
	if x_units is not None and x_units != '':
		y_label = f'{y_label} [{x_units}]'

	y_axis_params['axis_label'] = y_label

	clean_ax(ax, x_axis_params, y_axis_params)
	set_ax_limits(ax, x_axis_params, y_axis_params, allow_maxMult=True)

	if len(leg_objects) > 0:
		leg = fig.legend(leg_objects, [ob.get_label() for ob in leg_objects], fontsize=18, bbox_to_anchor=(0.7, 0.65, 0.2, 0.2), loc='upper center', ncol=1, borderaxespad=0.)
		leg.get_frame().set_edgecolor('none')
		leg.get_frame().set_facecolor('none')

	ann_texts.append({'label':ann_text_std(dt_start, dt_stop, ann_text_std_add), 'ha':'center'})
	ann_and_save(fig, ann_texts, inline, m_path, fname, tag)

########################################################
def plot_2d_hist(x_vals, y_vals, m_path, fname='2dhist', tag='', dt_start=None, dt_stop=None, inline=False, ann_text_std_add=None, ann_texts_in=None, binning={'x': {'nbins': None, 'min': None, 'max': None}, 'y': {'nbins': None, 'min': None, 'max': None}}, x_axis_params=None, y_axis_params=None, z_axis_params=None, show_bin_size=True, bin_size_str_fmt=None): # , weights=None
# x_axis_params={'axis_label': None, 'min': None, 'max': None, 'units': '', 'log': False}
# y_axis_params={'axis_label': None, 'min': None, 'max': None, 'units': '', 'log': False}
# z_axis_params={'axis_label': None, 'min': None, 'max': None, 'norm': None}

	ann_texts, x_axis_params, y_axis_params = _setup_vars(ann_texts_in, x_axis_params, y_axis_params)
	if not isinstance(z_axis_params, dict):
		z_axis_params = dict()

	norm = z_axis_params.get('norm', None)
	if norm == 'log':
		norm = LogNorm()
	elif norm is not None:
		raise ValueError('Unknown Norm!')

	fig, ax = plt.subplots(num=fname)
	fig.set_size_inches(aspect_ratio_single*vsize, vsize)

	h, x_edges, y_edges, image = ax.hist2d(x_vals, y_vals, bins=[binning.get('x', {}).get('nbins', None), binning.get('y', {}).get('nbins', None)],
range=[[binning.get('x', {}).get('min', None), binning.get('x', {}).get('max', None)], [binning.get('y', {}).get('min', None), binning.get('y', {}).get('max', None)]], cmin=z_axis_params.get('min', None), cmax=z_axis_params.get('max', None), norm=norm) # , weight=weights

	x_label = x_axis_params.get('axis_label', None)
	y_label = y_axis_params.get('axis_label', None)
	z_label = z_axis_params.get('axis_label', 'Counts')

	bin_size_ann = ''
	if show_bin_size:
		if bin_size_str_fmt is None:
			bin_size_str_fm = '.2f'

		x_bin_size = (max(x_edges)-min(x_edges))/(len(x_edges)-1)
		bin_size_ann = f' / {x_bin_size:{bin_size_str_fmt}}'

		x_units = x_axis_params.get('units', '')
		if x_units is not None and x_units != '':
			bin_size_ann = f'{bin_size_ann} {x_units}'

		y_bin_size = (max(y_edges)-min(y_edges))/(len(y_edges)-1)
		bin_size_ann = f'{bin_size_ann} x {y_bin_size:{bin_size_str_fmt}}'

		y_units = y_axis_params.get('units', '')
		if y_units is not None and y_units != '':
			bin_size_ann = f'{bin_size_ann} {y_units}'

	x_axis_params['axis_label'] = x_label
	y_axis_params['axis_label'] = y_label

	fig.colorbar(image, ax=ax, label=f'{z_label}{bin_size_ann}');

	clean_ax(ax, x_axis_params, y_axis_params)
	set_ax_limits(ax, x_axis_params, y_axis_params)

	ann_texts.append({'label':ann_text_std(dt_start, dt_stop, ann_text_std_add), 'ha':'center'})
	ann_and_save(fig, ann_texts, inline, m_path, fname, tag, ann_text_origin_x=std_ann_x-0.12)

########################################################
def plot_corr_matrix(dfp_corr, m_path, fname='correlation', tag='', dt_start=None, dt_stop=None, inline=False, ann_text_std_add=None, ann_texts_in=None, label_values=True, rotate_x_tick_labels=True):
	# create needed numpy objects from input dfp_OL
	invalid_val = -1.

	col_names = dfp_corr.columns
	num_vars = len(col_names)
	corr_matrix = dfp_corr.values

	mask_dups = np.zeros_like(corr_matrix, dtype=np.bool)
	mask_dups[np.triu_indices_from(mask_dups)]= True

	# mask_uncalculated = np.where(OLs == invalid_val, 1, 0)
	# mask = mask_dups | mask_uncalculated
	mask = mask_dups

	corr_matrix_masked = np.ma.masked_array(corr_matrix, mask=mask)

	# do the actual plotting
	ann_texts = []
	if ann_texts_in is not None:
		ann_texts = list(ann_texts_in)

	fig = plt.figure(fname)
	fig.set_size_inches(aspect_ratio_single*vsize*1.2, vsize*1.2)
	gs = gridspec.GridSpec(1,2, width_ratios=[20, 1])
	ax_left = plt.subplot(gs[0])

	norm = mpl.colors.Normalize(vmin=-1.,vmax=1.)
	cmap = cm.bwr
	img = ax_left.imshow(corr_matrix_masked, cmap=cmap, norm=norm, aspect='equal')
	# ax_left.set_aspect(1.)

	# annotate
	if label_values:
		for (j,i),value in np.ndenumerate(corr_matrix):
			if i < j and value!= invalid_val:
				# ax_left.text(i,j,f'{i},{j}', ha='center', va='center', color='fuchsia', size=10) # for debugging
				ax_left.text(i,j,f'{value:.2f}', ha='center', va='center', color='black', size=10)

	# cleanup
	ax_left.set_xticks(np.arange(num_vars))
	ax_left.set_yticks(np.arange(num_vars))

	if rotate_x_tick_labels:
		ax_left.set_xticklabels(col_names, rotation='vertical')
	else:
		ax_left.set_xticklabels(col_names)

	ax_left.set_yticklabels(col_names)

	# color bar
	cax = plt.subplot(gs[1])
	cb = plt.colorbar(img, cmap=cmap, norm=norm, cax=cax, label='Pearson Correlation')

	# cleanup (wasn't working when called earlier in function)
	ax_left.minorticks_off()

	ann_texts.append({'label':ann_text_std(dt_start, dt_stop, ann_text_std_add), 'ha':'center'})
	ann_and_save(fig, ann_texts, inline, m_path, fname, tag, ann_text_origin_x=std_ann_x-0.08, ann_text_origin_y=0.96, forced_text_size=12)

########################################################
# plot overlaid roc curves for many models
def plot_rocs(models, m_path, fname='roc', tag='', dt_start=None, dt_stop=None, inline=False, ann_text_std_add=None, ann_texts_in=None, rndGuess=False, better_ann=True, grid=False, inverse_log=False, precision_recall=False, pop_PPV=None, x_axis_params=None, y_axis_params=None):

	ann_texts, x_axis_params, y_axis_params = _setup_vars(ann_texts_in, x_axis_params, y_axis_params)

	fig, ax = plt.subplots()

	x_var = 'fpr'
	y_var = 'tpr'
	if precision_recall:
		x_var = 'rec'
		y_var = 'pre'

	if fname == 'roc':
		if precision_recall:
			fname = f'{fname}_precision_recall'
		if inverse_log:
			fname = f'{fname}_inverse_log'

	for model in models:
		# models is a list of dicts with keys name, nname, fpr, tpr, (or pre, rec), c (color), ls (linestyle)

		if inverse_log:
			with np.errstate(divide='ignore'):
				y_values = np.divide(1., model[y_var])
		else:
			y_values = model[y_var]

		label=f"{model['nname']}, AUC: {auc(model[x_var],model[y_var]):.4f}"

		ax.plot(model[x_var], y_values, lw=2, c=model.get('c', 'blue'), ls=model.get('ls', '-'), label=label)

		fname = f"{fname}_{model['name']}"

	if grid:
		ax.grid()

	if inverse_log:
		leg_loc = 'upper right'
	else:
		leg_loc = 'lower right'

	if rndGuess:
		if not precision_recall:
			if inverse_log:
				x = np.linspace(1e-10, 1.)
				ax.plot(x, 1/x, color='grey', linestyle=':', linewidth=2, label='Random Guess')
			else:
				x = np.linspace(0., 1.)
				ax.plot(x, x, color='grey', linestyle=':', linewidth=2, label='Random Guess')
		else:
			if pop_PPV is None:
				raise ValueError('Need pop_PPV to plot random guess curve for precision_recall!')
			if inverse_log:
				x = np.linspace(1e-10, 1.)
				y = pop_PPV*np.ones(len(x))
				ax.plot(x, 1/y, color='grey', linestyle=':', linewidth=2, label=f'Random Guess, PPV = {pop_PPV:.2f}')
			else:
				x = np.linspace(0., 1.)
				y = pop_PPV*np.ones(len(x))
				ax.plot(x, y, color='grey', linestyle=':', linewidth=2, label=f'Random Guess, PPV = {pop_PPV:.2f}')

	leg = ax.legend(loc=leg_loc,frameon=False)
	leg.get_frame().set_facecolor('none')

	xlabel = 'False Positive Rate'
	ylabel = 'True Positive Rate'
	if precision_recall:
		xlabel = 'Recall (Sensitivity, TPR)'
		ylabel = 'Precision (PPV)'

	ax.set_xlim([0.,1.])
	ax.set_xlabel(xlabel)
	if inverse_log:
		ax.set_yscale('log')
		ax.set_ylabel(f'Inverse {ylabel}')
	else:
		ax.set_ylim([0.,1.])
		ax.set_ylabel(ylabel)

	set_ax_limits(ax, x_axis_params, y_axis_params)

	if better_ann:
		if not precision_recall:
			if inverse_log:
				plt.text(-0.07, -0.12, 'Better', size=12, rotation=-45, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='green', alpha=0.2))
			else:
				plt.text(-0.07, 1.08, 'Better', size=12, rotation=45, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='green', alpha=0.2))
		else:
			if inverse_log:
				plt.text(1.07, -0.12, 'Better', size=12, rotation=45, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='green', alpha=0.2))
			else:
				plt.text(1.07, 1.08, 'Better', size=12, rotation=-45, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='green', alpha=0.2))

	ann_texts.append({'label':ann_text_std(dt_start, dt_stop, ann_text_std_add), 'ha':'center'})
	ann_and_save(fig, ann_texts, inline, m_path, fname, tag, ann_text_origin_x=std_ann_x-0.08, ann_text_origin_y=0.96, forced_text_size=12)

########################################################
# plot loss(es) vs epoch
def plot_loss_vs_epoch(dfp_train_results, m_path, fname='loss_vs_epoch', tag='', dt_start=None, dt_stop=None, inline=False, ann_text_std_add=None, ann_texts_in=None, x_axis_params=None, y_axis_params=None, loss_cols=['train_loss', 'val_loss'], std_ax_labels=True):

	ann_texts, x_axis_params, y_axis_params = _setup_vars(ann_texts_in, x_axis_params, y_axis_params)
	if std_ax_labels:
		x_axis_params['axis_label'] = 'Epoch'
		y_axis_params['axis_label'] = 'Loss'

	fig, ax = plt.subplots(num=fname)

	# hard coded, but should be fine
	loss_styles = {'train_loss': {'label': 'Training Loss', 'c': 'C0', 'ls': '--'},
						'val_loss': {'label': 'Validation Loss', 'c': 'C1', 'ls': ':'}
						}
	missing_loss_style = {'label': 'MISSING', 'c': 'red', 'ls': '-'}

	x_values = dfp_train_results['epoch'].values
	for loss_col in loss_cols:
		y_values = dfp_train_results[loss_col].values
		loss_style = loss_styles.get(loss_col, missing_loss_style)

		ax.plot(x_values, y_values, lw=2, label=loss_style['label'], c=loss_style['c'], ls=loss_style['ls'])

	leg = ax.legend(loc='upper right',frameon=False)
	leg.get_frame().set_facecolor('none')

	clean_ax(ax, x_axis_params, y_axis_params)
	set_ax_limits(ax, x_axis_params, y_axis_params)

	ann_texts.append({'label':ann_text_std(dt_start, dt_stop, ann_text_std_add), 'ha':'center'})
	ann_and_save(fig, ann_texts, inline, m_path, fname, tag)

########################################################
def plot_im(im, m_path, fname='im', tag='', dt_start=None, dt_stop=None, inline=False, ann_text_std_add=None, ann_texts_in=None, mean_unnormalize=None, std_unnormalize=None, im_vsize=4, turn_off_axes=True, x_axis_params=None, y_axis_params=None):
# x_axis_params={'axis_label':None, 'min':None, 'max':None}, y_axis_params={'axis_label':None, 'min':None, 'max':None}
	if not isinstance(im, np.ndarray):
		raise ValueError('Can not plot {type(im)}, convert to numpy array prior to plotting!')

	ann_texts, x_axis_params, y_axis_params = _setup_vars(ann_texts_in, x_axis_params, y_axis_params)

	fig, ax = plt.subplots(num=fname)
	if im_vsize is not None:
		fig.set_size_inches(im_vsize, im_vsize)
	else:
		fig.set_size_inches(aspect_ratio_single*vsize, vsize)

	# unnormalize and transpose from (channels, im_res, im_res) to (im_res, im_res, channels) for imshow plotting
	im = unnormalize_im(im, std_unnormalize, mean_unnormalize)
	im = np.transpose(im, (1, 2, 0))

	# plot imagee
	ax.imshow(im, aspect='equal')

	clean_ax(ax, x_axis_params, y_axis_params, turn_off_axes)
	set_ax_limits(ax, x_axis_params, y_axis_params)

	ann_texts.append({'label':ann_text_std(dt_start, dt_stop, ann_text_std_add), 'ha':'center'})
	ann_and_save(fig, ann_texts, inline, m_path, fname, tag)

########################################################
def plot_im_comp(im_left, im_right, m_path, fname='im_comp', tag='', dt_start=None, dt_stop=None, inline=False, ann_text_std_add=None, ann_texts_in=None, mean_unnormalize=None, std_unnormalize=None, im_vsize=4, turn_off_axes=True, x_axis_params=None, y_axis_params=None, ann_margin=True, left_right_orig_pred=True):
	# x_axis_params={'axis_label':None, 'min':None, 'max':None}, y_axis_params={'axis_label':None, 'min':None, 'max':None}
	if not isinstance(im_left, np.ndarray) or not isinstance(im_right, np.ndarray):
		raise ValueError('Can not plot {type(im)}, convert to numpy array prior to plotting!')

	ann_texts, x_axis_params, y_axis_params = _setup_vars(ann_texts_in, x_axis_params, y_axis_params)

	fig = plt.figure(fname)
	if im_vsize is not None:
		fig.set_size_inches(2*im_vsize, im_vsize)
	else:
		fig.set_size_inches(2*aspect_ratio_single*vsize, vsize)

	if ann_margin:
		gs = gridspec.GridSpec(1,3, width_ratios=[2, 2, 1])
		ax_ann_margin = plt.subplot(gs[2])
		ax_ann_margin.axis('off')
		ann_text_origin_x=std_ann_x+0.08
		ann_text_origin_y=std_ann_y-0.09
		forced_text_size = 12
	else:
		gs = gridspec.GridSpec(1,2, width_ratios=[1, 1])
		ann_text_origin_x=std_ann_x
		ann_text_origin_y=std_ann_y
		forced_text_size=None
	ax_left = plt.subplot(gs[0])
	ax_right = plt.subplot(gs[1])

	# unnormalize and transpose from (channels, im_res, im_res) to (im_res, im_res, channels) for imshow plotting
	im_left = unnormalize_im(im_left, std_unnormalize, mean_unnormalize)
	im_left = np.transpose(im_left, (1, 2, 0))
	im_right = unnormalize_im(im_right, std_unnormalize, mean_unnormalize)
	im_right = np.transpose(im_right, (1, 2, 0))

	# plot imagee
	ax_left.imshow(im_left, aspect='equal')
	ax_right.imshow(im_right, aspect='equal')

	for _ax in [ax_left, ax_right]:
		clean_ax(_ax, x_axis_params, y_axis_params, turn_off_axes)
		set_ax_limits(_ax, x_axis_params, y_axis_params)

	if left_right_orig_pred:
		if ann_margin:
			# TODO dial in x pos
			plt.figtext(0.2, 0.1, 'Original', ha='center', va='top', size=14, backgroundcolor='white')
			plt.figtext(0.59, 0.1, 'Predicted', ha='center', va='top', size=14, backgroundcolor='white')
		else:
			plt.figtext(0.25, 0.95, 'Original', ha='center', va='top', size=14, backgroundcolor='white')
			plt.figtext(0.75, 0.95, 'Predicted', ha='center', va='top', size=14, backgroundcolor='white')

	ann_texts.append({'label':ann_text_std(dt_start, dt_stop, ann_text_std_add), 'ha':'center'})
	ann_and_save(fig, ann_texts, inline, m_path, fname, tag, ann_text_origin_x, ann_text_origin_y, forced_text_size)
