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

plot_png=True
png_dpi=200

plot_jpg=False
jpg_dpi=200

########################################################
def plot_waveform(df, channel_names, m_path='output', fname='waveform', tag='', inline=False):

	fig, axs = plt.subplots(len(channel_names), sharex=True, sharey=True)

	this_vsize = 20
	this_aspect_ratio = 10. / 12. # width / height
	fig.set_size_inches(this_aspect_ratio*this_vsize, this_vsize)

	x = np.linspace(0., 1., len(df.index))

	for ichannel,channel_name in enumerate(channel_names):
		axs[ichannel].plot(x, df[channel_name], c=colors[ ichannel % len(colors) ])
		# axs[ichannel].axis('off')

		axs[ichannel].xaxis.set_ticklabels([])
		axs[ichannel].set_xlabel('')

		# axs[ichannel].yaxis.set_ticklabels([])
		axs[ichannel].set_ylabel(channel_name)

		# axs[ichannel].set_ylim([0.,1.])
	plt.tight_layout()
	if inline:
		fig.show()
	else:
		os.makedirs(m_path, exist_ok=True)
		if plot_png:
			fig.savefig(f'{m_path}/{fname}{tag}.png', dpi=png_dpi)
		if plot_jpg:
			fig.savefig(f'{m_path}/{fname}{tag}.jpg', dpi=jpg_dpi)
		plt.close('all')
