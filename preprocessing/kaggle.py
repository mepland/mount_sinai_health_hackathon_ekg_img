# Kaggle Waveform Preprocessing

#######################################################
# python
import os
import pandas as pd
import numpy as np
import tqdm

########################################################
# plotting
import matplotlib as mpl
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

########################################################
# load dfp from csv, clean up cols
def load_dfp(m_path, fname, tag='', debug=False, cols_str=[], cols_int=[], infer_header=True):
	full_fname = f'{m_path}/{fname}{tag}.csv'
	try:
		if debug:
			print('Attempting to open {0:s}'.format(full_fname))
		header = 'infer'
		if not infer_header:
			header = None
		dfp = pd.read_csv(full_fname, header=header)
		for col in dfp.columns:
			if col in cols_str:
				dfp[col] = dfp[col].astype(str)
			elif col in cols_int:
				dfp[col] = dfp[col].astype(int)
			else:
				dfp[col] = dfp[col].astype(float)

		return dfp
	except:
		if debug:
			raise ValueError('Could not open csv!')
		return None

########################################################

kaggle_data_path = '../data/kaggle_ds'

time_period = 1.5 # seconds
# sampling_rate = 125 # samples / second (Hz)

class_col = 187 # class stored in last column

classes_dict = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}
classes = sorted(list(classes_dict.keys()))

df_train = load_dfp(kaggle_data_path, 'mitbih_train', debug=False, infer_header=False, cols_int=[class_col])
df_test = load_dfp(kaggle_data_path, 'mitbih_test', debug=False, infer_header=False, cols_int=[class_col])
df = pd.concat([df_train, df_test])

########################################################

target_res = 400
size_in = 12
png_dpi = target_res/size_in
target_period=1.5
x = np.linspace(0., float(target_period), class_col-1)

########################################################
def plot_kaggle_waveform(y_array, m_path, fname):
	fig, ax = plt.subplots(num=fname)
	fig.set_size_inches(size_in, size_in)

	ax.plot(x, y_array, c='C0')
	ax.grid(which='both', axis='both')
	ax.set_ylabel('i')
	ax.set_xlabel('Time [S]')
	ax.set_xlim([0.,time_period])

	plt.tight_layout()
	fig.savefig(f'{m_path}/{fname}.png', dpi=png_dpi)
	plt.close('all')

########################################################
# run it

# TODO thread by hand!!!
_class = 0 # 0 through 4

class_str = classes_dict[_class]
print(f'Processing class: {class_str}')

df_class = df[df[class_col] == _class][list(range(class_col-1))]

class_path = f'./output/kaggle/{class_str}'
os.makedirs(class_path, exist_ok=True)

for index, row in tqdm.tqdm(df_class.iterrows(), total=df_class.shape[0]):
	plot_kaggle_waveform(row, class_path, f'wf_{index}')
