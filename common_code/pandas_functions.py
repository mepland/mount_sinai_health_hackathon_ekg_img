########################################################
# load package wide variables
from .configs import *

########################################################
def massage_dfp(dfp, target_fixed_cols=None, sort_by=None, sort_by_ascending=None):
	# sort rows
	if sort_by is not None:
		if isinstance(sort_by, str):
			sort_by = [sort_by]
		elif not isinstance(sort_by, list):
				raise TypeError(f'sort_by = {str(sort_by)} should be a str or list!')

		if sort_by_ascending is None:
			sort_by_ascending = True

		if isinstance(sort_by_ascending, bool):
			sort_by_ascending_arg = [sort_by_ascending for i in range(len(sort_by))]
		elif isinstance(sort_by_ascending, list):
			sort_by_ascending_arg = list(sort_by_ascending)
		else:
			raise TypeError(f'Unknown sort_by_ascending = {str(sort_by_ascending)}, should be a bool or list!')

		dfp = dfp.sort_values(by=sort_by, ascending=sort_by_ascending_arg).reset_index(drop=True)

	# order columns
	if target_fixed_cols is not None:
		cols = dfp.columns
		fixed_cols = [col for col in target_fixed_cols if col in cols]
		non_fixed_cols = list(set(cols) - set(fixed_cols))
		cols = list(fixed_cols + sorted(non_fixed_cols))
		dfp = dfp[cols]

	return dfp

########################################################
# create df from rows
def create_dfp(row_dicts, target_fixed_cols=None, sort_by=None, sort_by_ascending=None):
	# row_dicts = [{'col1':value {'col2':value ...] is a list of dicts, each becoming a row
	dfp = pd.DataFrame(row_dicts)
	dfp = massage_dfp(dfp, target_fixed_cols, sort_by, sort_by_ascending)

	return dfp

########################################################
# write out nice dfs to csv, creating them from rows first if necessary
def write_dfp(dfp_or_row_dicts, m_path, fname, tag='', target_fixed_cols=None, sort_by=None, sort_by_ascending=None, to_excel=False, to_html=False):

	if isinstance(dfp_or_row_dicts, pd.DataFrame):
		# dfp_or_row_dicts = is already a dfp
		dfp = dfp_or_row_dicts
	elif isinstance(dfp_or_row_dicts, list):
		# dfp_or_row_dicts = [{'col1':value {'col2':value ...] is a list of dicts, each becoming a row
	   dfp = pd.DataFrame(dfp_or_row_dicts)
	else:
		raise TypeError('Can only handle dfp or list or dict rows!!')

	dfp = massage_dfp(dfp, target_fixed_cols, sort_by, sort_by_ascending)

	# save
	os.makedirs(m_path, exist_ok=True)
	if to_excel:
		dfp.to_excel(f'{m_path}/{fname}{tag}.xlsx')
	elif to_html:
		dfp.to_html(f'{m_path}/{fname}{tag}.html')
	else:
		dfp.to_csv(f'{m_path}/{fname}{tag}.csv', index=False, na_rep='nan')

########################################################
# load dfp from csv, clean up cols
def load_dfp(m_path, fname, tag='', debug=False, cols_str=[], cols_float=[], cols_dt=[], cols_bool=[]):
	full_fname = f'{m_path}/{fname}{tag}.csv'

	try:
		if debug:
			print(f'Attempting to open {full_fname}')

		# keep leading 0s when loading strings
		# https://stackoverflow.com/questions/13250046/how-to-keep-leading-zeros-in-a-column-when-reading-csv-with-pandas/27144549#27144549
		converters = {}
		for col in cols_str:
			converters[col] = lambda x: str(x)

		dfp = pd.read_csv(full_fname, converters=converters)

		for col in dfp.columns:
			if debug:
				print(f'Changing dtype for col {col}')
			if col in cols_str:
				dfp[col] = dfp[col].astype(str)
			elif col in cols_float:
				dfp[col] = dfp[col].astype(float)
			elif col in cols_dt:
				dfp[col] = pd.to_datetime(dfp[col])
			elif col in cols_bool:
				dfp[col] = dfp[col].astype(bool)
			else:
				dfp[col] = dfp[col].astype(int)

		return dfp

	except:
		if debug:
			raise ValueError('Could not open csv!')
		return None

########################################################
def make_per_cols(dfp, num_col, denom_col, n_prefix='n_', per_prefix='per_'):
	per_col = num_col.replace(n_prefix, per_prefix)

	dfp[per_col] = dfp[num_col] / dfp[denom_col]
	dfp.loc[~np.isfinite(dfp[per_col]), per_col] = np.nan # change divide by zero +/-inf to NaN

	return dfp
