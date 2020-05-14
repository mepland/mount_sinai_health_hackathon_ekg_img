# python
import os
import pandas as pd
import numpy as np
# import math
# from collections import OrderedDict

########################################################
# plotting
import matplotlib as mpl
# mpl.use('Agg', warn=False)
# mpl.rcParams['font.family'] = ['HelveticaNeue-Light', 'Helvetica Neue Light', 'Helvetica Neue', 'Helvetica', 'Arial', 'Lucida Grande', 'sans-serif']
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.top']           = True
mpl.rcParams['ytick.right']         = True
mpl.rcParams['xtick.direction']     = 'in'
mpl.rcParams['ytick.direction']     = 'in'
mpl.rcParams['xtick.labelsize']     = 13
mpl.rcParams['ytick.labelsize']     = 13
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['xtick.major.width']   = 1.  # major tick width in points
mpl.rcParams['xtick.minor.width']   = 0.8  # minor tick width in points
mpl.rcParams['xtick.major.size']    = 7.0  # major tick size in points
mpl.rcParams['xtick.minor.size']    = 4.0  # minor tick size in points
mpl.rcParams['xtick.major.pad']     = 1.5  # distance to major tick label in points
mpl.rcParams['xtick.minor.pad']     = 1.4  # distance to the minor tick label in points
mpl.rcParams['ytick.major.width']   = 1.  # major tick width in points
mpl.rcParams['ytick.minor.width']   = 0.8  # minor tick width in points
mpl.rcParams['ytick.major.size']    = 7.0  # major tick size in points
mpl.rcParams['ytick.minor.size']    = 4.0  # minor tick size in points
mpl.rcParams['ytick.major.pad']     = 1.5  # distance to major tick label in points
mpl.rcParams['ytick.minor.pad']     = 1.4  # distance to the minor tick label in points
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from matplotlib import gridspec
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import matplotlib.ticker as ticker
# from matplotlib.ticker import LogLocator

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

########################################################
# Set common plot parameters
# vsize = 11 # inches
# aspect ratio width / height
# aspect_ratio_single = 4./3.
# aspect_ratio_multi = 1.

# Move here since these don't change per call of plot_waveform
size_in = 20

major_x = 0.2  # seconds
minor_x = 0.04 # seconds
major_y = 0.5 # mV
minor_y = 0.1 # mV

y_min_fixed = -2 # mV
y_max_fixed =  2 # mV

y_major_ticks_fixed = np.linspace(y_min_fixed, y_max_fixed, int((y_max_fixed-y_min_fixed)/major_y)+1)
y_minor_ticks_fixed = np.linspace(y_min_fixed, y_max_fixed, int((y_max_fixed-y_min_fixed)/minor_y)+1)

########################################################
def plot_waveform(dfp, channel_names, sampling_freq, m_path='output', fname='waveform', tag='', inline=False, target_time_range=5, target_im_res=800, run_parallel=False, fixed_yaxis_range=False, show_y_minor_grid=True):
	# setup

	# target_im_res = 1200 # decent quality
	# target_im_res = 800 # hackathon setting
	png_dpi = target_im_res/size_in

	n_channels = len(channel_names)
	n_samples = len(dfp.index)

	fig, axs = plt.subplots(n_channels, sharex=True, sharey=fixed_yaxis_range, num=fname)

	# this_vsize = 20
	# this_aspect_ratio = 10. / 12. # width / height
	# fig.set_size_inches(this_aspect_ratio*this_vsize, this_vsize)

	# fig.set_size_inches(target_im_res / png_dpi, target_im_res / png_dpi)
	fig.set_size_inches(size_in, size_in)

	if not run_parallel:
		data_time_range = n_samples / sampling_freq
		if target_time_range <= data_time_range:
			# we have enough samples to fill the requested target_time_range
			time_range = target_time_range
		else:
			# we do NOT have enough samples to fill the requested target_time_range, use all of the data we have
			time_range = data_time_range

		n_samples_to_use = int(np.floor(sampling_freq*time_range))

	else:
		# just set with out checking, would rather have the crash and debug
		time_range = target_time_range
		n_samples_to_use = n_samples

	x = np.linspace(0., float(time_range), n_samples_to_use)

	# start plotting

	for ichannel,channel_name in enumerate(channel_names):
		if not run_parallel:
			axs[ichannel].plot(x, dfp[channel_name].iloc[0:n_samples_to_use], c=colors[ ichannel % len(colors) ])
		else:
			axs[ichannel].plot(x, dfp[channel_name], c=colors[ ichannel % len(colors) ])

		axs[ichannel].set_ylabel(f'{channel_name} ', rotation='horizontal', ha='right')
		if ichannel == n_channels - 1:
			axs[ichannel].set_xlabel('Time [S]')

		axs[ichannel].grid(which='major', axis='both', color='#CCCCCC', alpha=1., lw=1)
		axs[ichannel].grid(which='minor', axis='both', color='#CCCCCC', alpha=0.5, lw=0.6)

		axs[ichannel].xaxis.set_ticks_position('none')
		axs[ichannel].yaxis.set_ticks_position('none')

	# clean up x axis limits and ticks
	axs[0].set_xlim([0.,float(time_range)])

	n_ticks_major_x = int(np.ceil(time_range / major_x))
	x_major_ticks = np.linspace(0., time_range, n_ticks_major_x+1)

	n_ticks_minor_x = int(np.ceil(time_range / minor_x))
	x_minor_ticks = np.linspace(0., time_range, n_ticks_minor_x+1)

	axs[0].set_xticks(x_major_ticks)
	axs[0].set_xticks(x_minor_ticks, minor=True)

	# clean up y_axis
	if fixed_yaxis_range:
		# apply new ticks and y limits
		axs[0].set_ylim([y_min_fixed,y_max_fixed])
		axs[0].set_yticks(y_major_ticks_fixed)
		if show_y_minor_grid:
			axs[0].set_yticks(y_minor_ticks_fixed, minor=True)
	else:
		# auto scale the y axis limits. This takes some doing to force the minor and major ticks to have the right size and always start at zero

		# first find the automatic y max and min across all of the channels
		for ichannel in range(n_channels):
			_y_min_auto, _y_max_auto = axs[ichannel].get_ylim()
			if ichannel == 0 or _y_min_auto < y_min_auto:
				y_min_auto = _y_min_auto
			if ichannel == 0 or y_max_auto < _y_max_auto:
				y_max_auto = _y_max_auto

		# round max and min to a minor tick, then add one minor tick to each side to be safe
		y_min = minor_y * (round(y_min_auto / minor_y)-1)
		y_max = minor_y * (round(y_max_auto / minor_y)+1)

		if y_min < 0:
			int_ticks_major_y_min = int(np.ceil( y_min / major_y))
			int_ticks_minor_y_min = int(np.ceil( y_min / minor_y))
		else:
			int_ticks_major_y_min = int(np.floor(y_min / major_y))
			int_ticks_minor_y_min = int(np.floor(y_min / minor_y))

		if y_max < 0:
			int_ticks_major_y_max = int(np.ceil(y_max / major_y))
			int_ticks_minor_y_max = int(np.ceil(y_max / minor_y))
		else:
			int_ticks_major_y_max = int(np.floor( y_max / major_y))
			int_ticks_minor_y_max = int(np.floor( y_max / minor_y))

		y_major_ticks = np.linspace(major_y*int_ticks_major_y_min, major_y*int_ticks_major_y_max, int_ticks_major_y_max-int_ticks_major_y_min+1)
		y_minor_ticks = np.linspace(minor_y*int_ticks_minor_y_min, minor_y*int_ticks_minor_y_max, int_ticks_minor_y_max-int_ticks_minor_y_min+1)

		# apply new ticks and y limits
		for ichannel in range(n_channels):
			axs[ichannel].set_ylim([y_min,y_max])
			axs[ichannel].set_yticks(y_major_ticks)
			if show_y_minor_grid:
				axs[ichannel].set_yticks(y_minor_ticks, minor=True)

	# save out
	plt.tight_layout()
	if inline and not run_parallel:
		fig.show()
	else:
		if not run_parallel:
			os.makedirs(m_path, exist_ok=True)
			fig.savefig(f'{m_path}/{fname}{tag}.pdf')
			fig.savefig(f'{m_path}/{fname}{tag}.png', dpi=png_dpi)
		else:
			fig.savefig(f'{m_path}/{fname}.png', dpi=png_dpi)
		plt.close('all')
