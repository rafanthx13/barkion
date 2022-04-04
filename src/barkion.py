import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt

################################################

def reduce_mem_usage(df, verbose=True):
	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
	start_mem = df.memory_usage().sum() / 1024**2    
	for col in df.columns:
		col_type = df[col].dtypes
		if col_type in numerics:
			c_min = df[col].min()
			c_max = df[col].max()
			if str(col_type)[:3] == 'int':
				if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
					df[col] = df[col].astype(np.int8)
				elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
					df[col] = df[col].astype(np.int16)
				elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
					df[col] = df[col].astype(np.int32)
				elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
					df[col] = df[col].astype(np.int64)  
			else:
				if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
					df[col] = df[col].astype(np.float16)
				elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
					df[col] = df[col].astype(np.float32)
				else:
					df[col] = df[col].astype(np.float64)    
	end_mem = df.memory_usage().sum() / 1024**2
	if verbose: print('Mem. usage decreased of {:5.2f} Mb to {:5.2f} Mb ({:.1f}% reduction)'.format(
		start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
	return df



def time_spent(time0):
	t = time.time() - time0
	t_int, t_min = int(t) // 60, t % 60
	return '{} min {:6.3f} s'.format(t_int, t_min) if t_int != 0 else '{:.3f} s'.format(t_min)

def df_value_counts(df0):
	# Usage: df_value_counts(df)
	columns = df0.columns.tolist()
	for c in columns:
		print('c', c)
		dtype = df0[c].dtype
		print('Column: "{}" | Type: {}\n==> '.format(c, dtype), end='')
		print('NUMBER OF UNIQUE VALUES: {:,d}\n==> '.format(len(df0[c].unique())), end='')
		# String, Categorical Feature
		if(dtype in ['O', 'object']):
			df_cat = pd.concat([df0[c].value_counts(), df0[c].value_counts(normalize=True)], axis=1).reset_index()
			df_cat.columns = ['values', 'count', 'percentage']
			if(len(df_cat) < 50):
				print('CATEGORICAL VALUES: ', end='')
				for _, row in df_cat.iterrows():
					print('{} ({:,d} = {:.2%}) ; '.format(row['values'], row['count'], row['percentage']), end='')
			else:
				acount = 0
				print('SOME CAT VALUES: ', end='')
				for _, row in df_cat.iterrows():
					print('{} ({:,d} = {:.2%}) ; '.format(row['values'], row['count'], row['percentage']), end='')
					acount += 1
					if(acount > 15):
						break
		# Number Feature
		elif(dtype in ['int64','float64']):
			df_int = pd.concat([df[c].value_counts(), df[c].value_counts(normalize=True)], axis=1).reset_index()
			df_int.columns = ['values', 'count', 'percentage']
			text = 'BINARY CAT VALUES: ' if len(df0[c]) == 2 else 'NUMERIC VALUES: '            
			print(text, end='')
			if(len(df_int) < 25):
				# Number but Binary Cat Feature
				for _, row in df_int.iterrows():
					print('{} ({:,d} = {:.2%}) ; '.format(row['values'], int(row['count']), row['percentage']), end='')
			else:
				# Numeric Feature
				acount = 0
				for _, row in df_int.iterrows():
					print('{} ({:,d} = {:.2%}) ; '.format(row['values'], int(row['count']), row['percentage']), end='')
					acount += 1
					if(acount > 10):
						break
				# Statistic
				print('\n==> STATISTICS:\n     ==> | ', end='')
				describ = df0[c].describe()
				acount = 0
				for index, value in describ.iteritems():
					print('{}: {:,.3f} | '.format(index, value), end='')
					acount += 1
					if(acount == 4):
						print('\n     ==> | '.format(index, value), end='')
		print('\n')



def generate_columns_from_index(topnum):
	adict = {}
	for i in range(topnum):
		adict[i] = 'top' + str(i+1) + '°'
	return adict

def eda_categ_feat_desc_df(series_categorical):
	"""Generate DataFrame with quantity and percentage of categorical series
	@series_categorical = categorical series
	"""
	series_name = series_categorical.name
	val_counts = series_categorical.value_counts()
	val_counts.name = 'quantity'
	val_percentage = series_categorical.value_counts(normalize=True).apply(lambda x: '{:.2%}'.format(x))
	val_percentage.name = "percentage"
	val_concat = pd.concat([val_counts, val_percentage], axis = 1)
	val_concat.reset_index(level=0, inplace=True)
	val_concat = val_concat.rename( columns = {'index': series_name} )
	return val_concat

def eda_categ_feat_T_rankend(series, top_num):
	"""
	Usage: eda_categ_feat_T_rankend(df['acolumn'], 5)
	Show @top_num value in frequence to a @series
	"""
	return eda_categ_feat_desc_df(series).head(top_num).T.rename(generate_columns_from_index(top_num),axis='columns')

def describe_horizontal_serie(serie):
	"""
	describe_horizontal_serie(df['acolumn'])
	show table of df.descrbe() in other format
	"""
	adec = serie.describe()
	adtype = serie.dtype
	adf = pd.DataFrame(data=adec.values).T
	adf.columns = adec.index
	adf.index = pd.Index([adec.name])
	if(adtype in ['int64']):
		alist = ['min', '25%', '50%', '75%', 'max']
		for c in alist:
			adf[c] = adf[c].astype('int64')
			adf[c] = adf[c].map(lambda x: "{:,d}".format(int(x)))
	adf['count'] = adf['count'].map(lambda x: "{:,d}".format(int(x)))
	return adf

def eda_cat_top_slice_count(s, start=1, end=None, rotate=0):
	"""
	Create columns of series
	"""
	# @rotate: 45/80; 
	column, start, threshold = s.name, start - 1, 30
	s = df[column].value_counts()
	lenght = len(s)
	if(end is None):
		end = lenght if lenght <= threshold else threshold
	s = s.reset_index()[start:end]
	s = s.rename(columns = {column: 'count'}).rename(columns = {'index': column,})
	fig, ax = plt.subplots(figsize = (12,4))
	barplot = sns.barplot(x=s[column], y=s['count'], ax=ax)
	# sort by name
	s = s.sort_values(column).reset_index()
	for index, row in s.iterrows():
		barplot.text(row.name, row['count'], '{:,d}'.format(row['count']), color='black', ha="center")
	ax.set_title('Quantity Plot to {}. Top {}°-{}°'.format(column, start+1, end))
	plt.xticks(rotation=rotate)
	plt.show()
# eda_cat_top_slice_count(df['Year'], start=5, end=10, rotate=0)



def eda_categ_feat_desc_plot(series_categorical, title = "", fix_labels=False):
	"""Generate 2 plots: barplot with quantity and pieplot with percentage. 
	   @series_categorical: categorical series
	   @title: optional
	   @fix_labels: The labes plot in barplot in sorted by values, some times its bugs cuz axis ticks is alphabethic
		   if this happens, pass True in fix_labels
	   @bar_format: pass {:,.0f} to int
	"""
	series_name = series_categorical.name
	val_counts = series_categorical.value_counts()
	val_counts.name = 'quantity'
	val_percentage = series_categorical.value_counts(normalize=True)
	val_percentage.name = "percentage"
	val_concat = pd.concat([val_counts, val_percentage], axis = 1)
	val_concat.reset_index(level=0, inplace=True)
	val_concat = val_concat.rename( columns = {'index': series_name} )
	
	fig, ax = plt.subplots(figsize = (12,4), ncols=2, nrows=1) # figsize = (width, height)
	if(title != ""):
		fig.suptitle(title, fontsize=18)
		fig.subplots_adjust(top=0.8)

	s = sns.barplot(x=series_name, y='quantity', data=val_concat, ax=ax[0])
	if(fix_labels):
		val_concat = val_concat.sort_values(series_name).reset_index()
	
	for index, row in val_concat.iterrows():
		s.text(row.name, row['quantity'], '{:,d}'.format(int(row['quantity'])), color='black', ha="center")

	s2 = val_concat.plot.pie(y='percentage', autopct=lambda value: '{:.2f}%'.format(value),
							 labels=val_concat[series_name].tolist(), legend=None, ax=ax[1],
							 title="Percentage Plot")

	ax[1].set_ylabel('')
	ax[0].set_title('Quantity Plot')

	plt.show()


def eda_categ_feat_desc_df(series_categorical):
	"""Generate DataFrame with quantity and percentage of categorical series
	eda_horiz_plot(df_california.head(top_umber), 'count', 'city', 'Rank 10 City death in CA')
	@series_categorical = categorical series
	"""
	series_name = series_categorical.name
	val_counts = series_categorical.value_counts()
	val_counts.name = 'quantity'
	val_percentage = series_categorical.value_counts(normalize=True).apply(lambda x: '{:.2%}'.format(x))
	val_percentage.name = "percentage"
	val_concat = pd.concat([val_counts, val_percentage], axis = 1)
	val_concat.reset_index(level=0, inplace=True)
	val_concat = val_concat.rename( columns = {'index': series_name} )
	return val_concat

def eda_horiz_plot(df, x, y, title, figsize = (8,5), palette="Blues_d", formating="int"):
	"""Using Seaborn, plot horizonal Bar with labels
	!!! Is recomend sort_values(by, ascending) before passing dataframe
	!!! pass few values, not much than 20 is recommended
	"""
	f, ax = plt.subplots(figsize=figsize)
	sns.barplot(x=x, y=y, data=df, palette=palette)
	ax.set_title(title)
	for p in ax.patches:
		width = p.get_width()
		if(formating == "int"):
			text = int(width)
		else:
			text = '{.2f}'.format(width)
		ax.text(width + 1, p.get_y() + p.get_height() / 2, text, ha = 'left', va = 'center')
	plt.show()

# Example Using:
# top_umber = 10
# df_california = df.query("state == 'CA'").groupby(['city']).count()['date'].sort_values(ascending = False).reset_index().rename({'date': 'count'}, axis = 1)
# list_cities_CA = list(df_california.head(top_umber)['city']) 

def series_remove_outiliers(series):
	# Use IQR Strategy
	# https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/
	# def quantils
	q25, q75 = series.quantile(0.25), series.quantile(0.75)
	iqr = q75 - q25
	print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
	cut_off = iqr * 1.5
	lower, upper = q25 - cut_off, q75 + cut_off
	# identify outliers
	print('Cut Off: below than', lower, 'and above than', upper)
	outliers = series[ (series > upper) | (series < lower)]
	print('Identified outliers: {:,d}'.format(len(outliers)), 'that are',
		  '{:.2%}'.format(len(outliers)/len(series)), 'of total data')
	# remove outliers
	outliers_removed = [x for x in series if x >= lower and x <= upper]
	print('Non-outlier observations: {:,d}'.format(len(outliers_removed)))
	series_no_outiliers = series[ (series <= upper) & (series >= lower) ]
	return series_no_outiliers


def df_remove_outiliers_from_a_serie(mydf, series_name):
	# Use IQR Strategy
	# https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/
	# def quantils
	series = mydf[series_name]
	q25, q75 = series.quantile(0.25), series.quantile(0.75)
	iqr = q75 - q25
	print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
	cut_off = iqr * 1.5
	lower, upper = q25 - cut_off, q75 + cut_off
	# identify outliers
	print('Cut Off: below than', lower, 'and above than', upper)
	outliers = series[ (series > upper) | (series < lower)]
	print('Identified outliers: {:,d}'.format(len(outliers)), 'that are',
		  '{:.2%}'.format(len(outliers)/len(series)), 'of total data')
	# remove outliers
	outliers_removed = [x for x in series if x >= lower and x <= upper]
	print('Non-outlier observations: {:,d}'.format(len(outliers_removed)))
	mydf = mydf[ (mydf[series_name] <= upper) & (mydf[series_name] >= lower) ]
	return mydf



def find_outiliers_from_series(df_num): 
	""""must be float64 dtype"""
	
	# calculating mean and std of the array
	data_mean, data_std = np.mean(df_num), np.std(df_num)

	# seting the cut line to both higher and lower values
	# You can change this value
	cut = data_std * 3

	#Calculating the higher and lower cut values
	lower, upper = data_mean - cut, data_mean + cut

	# creating an array of lower, higher and total outlier values 
	outliers_lower = [x for x in df_num if x < lower]
	outliers_higher = [x for x in df_num if x > upper]
	outliers_total = [x for x in df_num if x < lower or x > upper]

	# array without outlier values
	outliers_removed = [x for x in df_num if x > lower and x < upper]
	
	print('OUTILIERS:\nmean: {:.4f} | std: {:.4f} |\nlower_cutter: {:.4f} | upper_cutter: {:.4f}'.format(data_mean, data_std, lower, upper))
	print('Identified lowest outliers: %d' % len(outliers_lower)) # printing total number of values in lower cut of outliers
	print('Identified upper outliers: %d' % len(outliers_higher)) # printing total number of values in higher cut of outliers
	print('Total outlier observations: %d' % len(outliers_total)) # printing total number of values outliers of both sides
	print('Non-outlier observations: %d' % len(outliers_removed)) # printing total number of non outlier values
	# Percentual of outliers in points
	print("Total percentual of Outliers: {:.2%}".format( len(outliers_total) / len(outliers_removed) ) ) 
	


def eda_numerical_feat(series, title="", with_label=True, number_format="", show_describe=False, size_labels=10):
	# Use 'series_remove_outiliers' to filter outiliers
	""" Generate series.describe(), bosplot and displot to a series
	@with_label: show labels in boxplot
	@number_format: 
		integer: 
			'{:d}'.format(42) => '42'
			'{:,d}'.format(12855787591251) => '12,855,787,591,251'
		float:
			'{:.0f}'.format(91.00000) => '91' # no decimal places
			'{:.2f}'.format(42.7668)  => '42.77' # two decimal places and round
			'{:,.4f}'.format(1285591251.78) => '1,285,591,251.7800'
			'{:.2%}'.format(0.09) => '9.00%' # Percentage Format
		string:
			ab = '$ {:,.4f}'.format(651.78) => '$ 651.7800'
	def swap(string, v1, v2):
		return string.replace(v1, "!").replace(v2, v1).replace('!',v2)
	# Using
		swap(ab, ',', '.')
	"""
	f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 5), sharex=False)
	if(show_describe):
		print(series.describe())
	if(title != ""):
		f.suptitle(title, fontsize=18)
	sns.distplot(series, ax=ax1)
	sns.boxplot(series, ax=ax2)
	if(with_label):
		describe = series.describe()
		labels = { 'min': describe.loc['min'], 'max': describe.loc['max'], 
			  'Q1': describe.loc['25%'], 'Q2': describe.loc['50%'],
			  'Q3': describe.loc['75%']}
		if(number_format != ""):
			for k, v in labels.items():
				ax2.text(v, 0.3, k + "\n" + number_format.format(v), ha='center', va='center', fontweight='bold',
						 size=size_labels, color='white', bbox=dict(facecolor='#445A64'))
		else:
			for k, v in labels.items():
				ax2.text(v, 0.3, k + "\n" + str(v), ha='center', va='center', fontweight='bold',
					 size=size_labels, color='white', bbox=dict(facecolor='#445A64'))
	plt.show()



def describe_y_classify_by_cat_feat(mydf, x, y, title='', classify_content='survivors', labels=['Death', 'Survived']):
	"""
	# func(df, x='embarked', y='survived', title='survived by sex')
	Generate one barplot with quantity and len(x.unique()) pie plots with percentage of x by class of y.unique()
	@classify_content : string that is the meaning of y
	@labels : start from 0, is the meanign of y value
	"""
	# Create DataSet
	df1 = df.groupby([x,y]).count().reset_index()
	a_column = df1.columns[2]
	df1 = df1.rename({a_column: "quantity"}, axis=1)
	alist = df1['quantity'].tolist()
	unique_values_x = mydf[x].unique().tolist()
	unique_values_x.sort()
	len_unique_values_y = len(mydf[y].unique().tolist())
	# Create Fig and Axes
	f, ax = plt.subplots(ncols=len(unique_values_x)+1, figsize=(18, 5), sharex=False)
	f.suptitle(title, fontsize=18)
	# BarPlot
	s = sns.barplot(x=x, y='quantity', hue=y, data=df1, ax=ax[0])
	count, by_hue = 0, 0
	for index, row in df1.iterrows():
		axis_x = count - 0.20 if index % 2 == 0 else count + 0.20
		by_hue += 1
		if(by_hue == len_unique_values_y):
			count += 1
			by_hue = 0
			# print(axis_x) ## DEBUG
		s.text(axis_x, row['quantity'], '{:,d}'.format(int(row['quantity'])), color='black', ha="center")
	# Query DF
	hue_count = 0
	for i in range(len(unique_values_x)):
		df1.query('{} == "{}"'.format(x, unique_values_x[i])).plot.pie(y='quantity', figsize=(18, 5), autopct='%1.2f%%',
									labels = ['{} = {}'.format(labels[0], str(alist[i+hue_count])),
											  '{} = {}'.format(labels[1], str(alist[i+hue_count+1]))],
									title='{} {} {} (Total = {})'.format(x, unique_values_x[i], classify_content ,str(alist[i] + alist[i+1])),
									ax=ax[i+1], labeldistance=None)
		hue_count += 1
	plt.show()
	# return df1 ## DEBUG
	

def describe_y_classify_numeric_feature(mydf, x, y, title='', with_swarmp=False):
	"""
	func(df, x='fare', y='survived')
	"""
	f, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18, 5), sharex=False)
	# Box and Violin Plots
	sns.boxplot(y=x, x=y, data=mydf, ax=ax1)
	sns.violinplot(y=x, x=y, data=mydf, ax=ax2)
	if(with_swarmp):
		sns.swarmplot(x=y, y=x, data=mydf, ax=ax2, palette='rocket')
	# HistogramPlot
	y_unique_values = mydf[y].unique().tolist()
	for u in y_unique_values:
		adf = mydf.query("{} == {}".format(y, u))
		sns.distplot(adf[x], ax=ax3)
	# Set Titles
	if(not title):
		f.suptitle('{} by {}'.format(y,x), fontsize=18)
	else:
		f.suptitle(title, fontsize=18)
	ax1.set_title("BoxPlot")
	ax2.set_title("ViolinPlot")
	ax3.set_title("HistogramPlot")
	plt.show()
	
def plot_top_rank_correlation(my_df, column_target):
	corr_matrix = my_df.corr()
	top_rank = len(corr_matrix)
	f, ax1 = plt.subplots(ncols=1, figsize=(18, 6), sharex=False)

	ax1.set_title('Top Correlations to {}'.format(top_rank, column_target))
	
	cols_top = corr_matrix.nlargest(len(corr_matrix), column_target)[column_target].index
	cm = np.corrcoef(my_df[cols_top].values.T)
	mask = np.zeros_like(cm)
	mask[np.triu_indices_from(mask)] = True
	hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
					 annot_kws={'size': 10}, yticklabels=cols_top.values,
					 xticklabels=cols_top.values, mask=mask, ax=ax1)
	
	plt.show()



def plot_top_bottom_rank_correlation(my_df, column_target, top_rank=5):
	corr_matrix = my_df.corr()
	f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 6), sharex=False)

	ax1.set_title('Top {} Positive Corr to {}'.format(top_rank, column_target))
	ax2.set_title('Top {} Negative Corr to {}'.format(top_rank, column_target))
	
	cols_top = corr_matrix.nlargest(top_rank+1, column_target)[column_target].index
	cm = np.corrcoef(my_df[cols_top].values.T)
	mask = np.zeros_like(cm)
	mask[np.triu_indices_from(mask)] = True
	hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
					 annot_kws={'size': 8}, yticklabels=cols_top.values,
					 xticklabels=cols_top.values, mask=mask, ax=ax1)
	
	cols_bot = corr_matrix.nsmallest(top_rank, column_target)[column_target].index
	cols_bot  = cols_bot.insert(0, column_target)
	print(cols_bot)
	cm = np.corrcoef(my_df[cols_bot].values.T)
	mask = np.zeros_like(cm)
	mask[np.triu_indices_from(mask)] = True
	hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
					 annot_kws={'size': 10}, yticklabels=cols_bot.values,
					 xticklabels=cols_bot.values, mask=mask, ax=ax2)
	
	plt.show()



from scipy.stats import norm
from scipy import stats

def test_normal_distribution(serie, thershold=0.4):
	series_name = serie.name
	f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 6), sharex=False)
	f.suptitle('{} is a Normal Distribution?'.format(series_name), fontsize=18)
	ax1.set_title("Histogram to " + series_name)
	ax2.set_title("Q-Q-Plot to "+ series_name)
	
	# calculate normal distrib. to series
	mu, sigma = norm.fit(serie)
	print('Normal dist. (mu= {:,.2f} and sigma= {:,.2f} )'.format(mu, sigma))
	
	# skewness and kurtoise
	skewness = serie.skew()
	kurtoise = serie.kurt()
	print("Skewness: {:,.2f} | Kurtosis: {:,.2f}".format(skewness, kurtoise))
	# evaluate skeness
	# If skewness is less than −1 or greater than +1, the distribution is highly skewed.
	# If skewness is between −1 and −½ or between +½ and +1, the distribution is moderately skewed.
	# If skewness is between −½ and +½, the distribution is approximately symmetric.
	pre_text = '\t=> '
	if(skewness < 0):
		text = pre_text + 'negatively skewed or left-skewed'
	else:
		text =  pre_text + 'positively skewed or right-skewed\n'
		text += pre_text + 'in case of positive skewness, log transformations usually works well.\n'
		text += pre_text + 'np.log(), np.log1(), boxcox1p()'
	if(skewness < -1 or skewness > 1):
		print("Evaluate skewness: highly skewed")
		print(text)
	if( (skewness <= -0.5 and skewness > -1) or (skewness >= 0.5 and skewness < 1)):
		print("Evaluate skewness: moderately skewed")
		print(text)
	if(skewness >= -0.5 and skewness <= 0.5):
		print('Evaluate skewness: approximately symmetric')
	# evaluate kurtoise
	#     Mesokurtic (Kurtoise next 3): This distribution has kurtosis statistic similar to that of the normal distribution.
	#         It means that the extreme values of the distribution are similar to that of a normal distribution characteristic. 
	#         This definition is used so that the standard normal distribution has a kurtosis of three.
	#     Leptokurtic (Kurtosis > 3): Distribution is longer, tails are fatter. 
	#         Peak is higher and sharper than Mesokurtic, which means that data are heavy-tailed or profusion of outliers.
	#         Outliers stretch the horizontal axis of the histogram graph, which makes the bulk of the data appear in a 
	#         narrow (“skinny”) vertical range, thereby giving the “skinniness” of a leptokurtic distribution.
	#     Platykurtic: (Kurtosis < 3): Distribution is shorter, tails are thinner than the normal distribution. The peak
	#         is lower and broader than Mesokurtic, which means that data are light-tailed or lack of outliers.
	#         The reason for this is because the extreme values are less than that of the normal distribution.
	print('evaluate kurtoise')
	if(kurtoise > 3 + thershold):
		print(pre_text + 'Leptokurtic: anormal: Peak is higher')
	elif(kurtoise < 3 - thershold):
		print(pre_text + 'Platykurtic: anormal: The peak is lower')
	else:
		print(pre_text + 'Mesokurtic: normal: the peack is normal')
	
	# shapiro-wilki test normality
	# If the P-Value of the Shapiro Wilk Test is larger than 0.05, we assume a normal distribution
	# If the P-Value of the Shapiro Wilk Test is smaller than 0.05, we do not assume a normal distribution
	#     print("Shapiro-Wiki Test: Is Normal Distribution? {}".format(stats.shapiro(serie)[1] < 0.01) )
	#     print(stats.shapiro(serie))

	
	# ax1 = histogram
	sns.distplot(serie , fit=norm, ax=ax1)
	ax1.legend(['Normal dist. ($\mu=$ {:,.2f} and $\sigma=$ {:,.2f} )'.format(mu, sigma)],
			loc='best')
	ax1.set_ylabel('Frequency')
	# ax2 = qq-plot
	stats.probplot(serie, plot=ax2)
	plt.show()



def df_rating_missing_data(my_df):
	"""Create DataFrame with Missing Rate
	"""
	# get sum missing rows and filter has mising values
	ms_sum = my_df.isnull().sum()
	ms_sum = ms_sum.drop( ms_sum[ms_sum == 0].index )
	# get percentage missing ratio and filter has mising values
	ms_per = (my_df.isnull().sum() / len(my_df))
	ms_per = ms_per.drop( ms_per[ms_per == 0].index)
	# order by
	ms_per = ms_per.sort_values(ascending=False)
	ms_sum = ms_sum.sort_values(ascending=False)
	# format percentage
	ms_per = ms_per.apply(lambda x: '{:.3%}'.format(x))
	return pd.DataFrame({'Missing Rate' : ms_per, 'Count Missing': ms_sum})  


def plot_model_score_regression(models_name_list, model_score_list, title=''):
	fig = plt.figure(figsize=(15, 6))
	ax = sns.pointplot( x = models_name_list, y = model_score_list, 
		markers=['o'], linestyles=['-'])
	for i, score in enumerate(model_score_list):
		ax.text(i, score + 0.002, '{:.4f}'.format(score),
				horizontalalignment='left', size='large', 
				color='black', weight='semibold')
	plt.ylabel('Score', size=20, labelpad=12)
	plt.xlabel('Model', size=20, labelpad=12)
	plt.tick_params(axis='x', labelsize=12)
	plt.tick_params(axis='y', labelsize=12)
	plt.xticks(rotation=70)
	plt.title(title, size=20)
	plt.show()


from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score

def evaluate_regression(y_pred, y_test, title=''):
	if(title):
		print(title)
	print('MAE  : {:14,.3f}'.format(mean_absolute_error(y_pred, y_test)))
	print('MSE  : {:14,.3f}'.format(mean_squared_error(y_pred, y_test)))
	print('RMSE : {:14,.3f}'.format(np.sqrt(mean_squared_error(y_pred, y_test))))
	print('RMSLE: {:14,.3f}'.format(np.sqrt(mean_squared_log_error(np.absolute(y_pred), y_test))))
	print('R2   : {:14,.3f}'.format(r2_score(y_pred, y_test)))




def describe_y_numeric_by_x_cat_boxplot(dtf, x_feat, y_target, title='', figsize=(15,5), rotatioon_degree=0):
	""" Generate a quickly boxplot  to describe each Ŷ by each categorical value of x_feat
	"""
	the_title = title if title != '' else '{} by {}'.format(y_target, x_feat)
	fig, ax1 = plt.subplots(figsize = figsize)
	sns.boxplot(x=x_feat, y=y_target, data=dtf, ax=ax1)
	ax1.set_title(the_title, fontsize=18)
	plt.xticks(rotation=rotatioon_degree)
	plt.show()
 # Example
 # describe_y_numeric_by_x_cat_boxplot(df, 'score', 'comment_len', figsize=(10,5))



def show_unique_values_by_percentage(series):
	"""Generate a string with resume of value counts with percentage
	Like: '[RECORRENTE(56.63%), POS-PAID(42.81%), PRE-PAID(0.55%), NENHUMA(0.00%)]'
	"""
	series_name = series.name
	val_counts = series.value_counts()
	val_counts.name = 'count'
	val_percentage = series.value_counts(
		normalize=True).apply(lambda x: '{:.2%}'.format(x))
	val_percentage.name = "percentage"
	val_concat = pd.concat([val_counts, val_percentage], axis=1)
	val_concat.reset_index(level=0, inplace=True)
	val_concat = val_concat.rename(columns={'index': series_name})
	astr = '['
	for index, row in val_concat.iterrows():
		astr += str(row[series_name]) + ' (' + str(row['percentage']) + '), '
	astr = astr[:-2]
	astr += ']'
	return astr



def df_delete_column_by_threshold_nan(adf, threshold=0.40):
	"""
	Delete colums from @adf where are @threshold of NaN
	Ex: 0.40 => Vai deletar todas as colunas se tiver com 40% 
	de dados válidos ou menos.
	"""
	# before
	adf_before = adf.columns.tolist()
	len_before = len(adf_before)
	# process
	adf = adf.loc[:, adf.isnull().mean() < 1 - threshold]
	# after
	adf_after = adf.columns.tolist()
	len_after = len(adf_after)
	diff = len_before - len_after
	# print
	print('Excluiu:', diff, 'colunas, que equivale a',
		  "{:3.2%}".format(diff/len_before),
		  'do total de',len_before ,'colunas originalmente',
		  '\n\nExcluiu as seguinte colunas:\n')
	alist = []
	for el in adf_before:
		if(el not in adf_after):
			alist.append(el)
	print(alist)
	return adf




def df_delete_colums_with_unique_values(adf, unique_values=1):
	"""
	Delete colunms of pd.Dataframe @adf if his 
	number of unique value is @unique_values or less
	"""
	# before
	adf_before = adf.columns.tolist()
	len_before = len(adf_before)
	# procedure
	list_to_exclude = [
		col for col in adf_before
		if len(adf[col].value_counts()) <= unique_values
	]
	adf = adf.drop(list_to_exclude, axis=1)
	# after
	adf_after = adf.columns.tolist()
	len_after = len(adf_after)
	diff = len_before - len_after
	# print
	print('Excluiu:', diff, 'colunas, que equivale a',
		  "{:3.2%}".format(diff/len_before ), 'do numero total de colunas', len_before)
	print('  (por ter a quantidade de valores unicos <= {}):\n'.format(unique_values))
	print(list_to_exclude)
	return adf

######################

# JOIN TOGETHER
def check_duplicate(series, qtd_rows=0):
	if(qtd_rows == 0):
		qtd_rows = len(series)
	value_unique = len(series.unique())
	valid_sum = series.value_counts().sum()
	count_dif_values = 'Count Different Values: ' + str(value_unique)
	if(np.NaN in series.unique().tolist()):
		count_dif_values += ' (includes Nan)'
	print(count_dif_values, '\n',
		  'Percentage of Unique Values:', "{:.3%}".format(
			  value_unique/qtd_rows), '\n',
		  'Percentage of Duplicate Values :', "{:.3%}".format(1 - (value_unique/qtd_rows)))
	print('\nValid and Invalid Values (np.NaN)')
	print('\n Valid Values   (!= NaN):', "{:8.5%}".format(valid_sum/qtd_rows),
		  '\n Invalid Values (== NaN):', "{:8.5%}".format((qtd_rows-valid_sum)/qtd_rows))

# Count Frequency with percentage
def series_value_counts(series):
	series_name = series.name
	val_counts = series.value_counts()
	val_counts.name = 'count'
	val_percentage = series.value_counts(
		normalize=True).apply(lambda x: '{:.2%}'.format(x))
	val_percentage.name = "percentage"
	val_concat = pd.concat([val_counts, val_percentage], axis=1)
	val_concat.reset_index(level=0, inplace=True)
	val_concat = val_concat.rename(columns={'index': series_name})
	display(val_concat)


def analysis_column(series, qtd_rows=0):
	print('========== {} ==========='.format(series.name))
	print('\n\tCount Unique and Duplicate Values\n')
	check_duplicate(series, qtd_rows)
	print('\n\tANALYSIS OF VALID VALUE COUNTING')
	series_value_counts(series)
	print()
 
#########################

def show_all_rows(df, column, value, show_df=True, g_file=False, file_path = 'example.csv'):
	"""Display and generate csv of a value of a column
		EXAMPLE show_all_rows(df_cliente, 'CUSTOMER_KEY', 10001032)
	"""
	if(file_path == ''):
		file_path = 'df_' + str(column) + '_' + str(value) + '.csv'
	df_of_value = df[ df[column] == value]
	if(show_df):
		display(df_of_value)
	if(g_file):
		if(not os.path.isfile(file_path)):
			print('Generate', file_path)
			df_of_value.to_csv(file_path, sep = ';', index = False) # usar separador ';' no libreofice
		else:
			print('pdf', file_path ,'ja gerado')
	else:
		print('Nao gerar arquivo')
	return df_of_value

def what_varies_in_a_df(df, show=5):
	#  what_column_is_dif what varies in a column
	"""To a df, show if values of columns repeat or not
			Utils with  return of 'show_all_rows'
			@show = qtd of value diff to show
		LIKE: what_column_is_dif(show_all_rows(df_cliente, 'CUSTOMER_KEY', 10001032))
	"""
	columns = df.columns.tolist()
	# show total rows
	print('QTD Rows', len(df), '\n')
	# get max len of columns
	max_length = 0
	for i in columns:
		if(len(i) > max_length):
			max_length = len(i)
	aformat = '{:<' + str(max_length) + '}'
	# show columns
	for c in columns:
		list_values = df[c].unique()
		count_uniques = len(list_values)
		if(count_uniques == 1):
			print((aformat + ' IS UNIQUE   {}').format(c, df[c].unique()))
		else:
			if(count_uniques > 10):
				# show diff values
				c_diff = len(df[c].unique())
				print(
					(aformat + ' IS DIFF IN  {:<5} == {:6.2%} OF TOTAL').format(c, c_diff, c_diff / len(df[c])))
			else:
				print((aformat + ' IS DIFF     {}').format(c, df[c].unique()))

def df_check_unique_values_of_columns(adf, return_df=False):
    """
    Conta a quantidade de valore súnicos para cada coluna
    """
    cols = adf.columns.tolist()
    # get max string
    formating_col = '{:<'+str(len((max(cols, key=len))))+'}'
    for col in cols:
        print(formating_col.format(col), '::', len(adf[col].value_counts()))
    if(return_df):
        indx, values = [], []
        for col in cols:
            indx.append(col)
            values.append(len(adf[col].value_counts()))
        n_df = pd.DataFrame(
            data=zip(indx,values),
            columns=['Column','Qtd_Unique_Values'])
        return n_df.sort_values('Qtd_Unique_Values')
    
def show_rows_duplicates_samples(series, df, repeat=1, top_elements=3):
    """Show 3 DF with rows of 3 most frequency values of a series"""
    print('Show {} rows of {} most frequency values of series'.format(
        repeat+1, top_elements), '\n')
    val_counts = series.value_counts()
    top_indexs = []
    for i in range(top_elements):
        top_indexs.append(val_counts.index[i])
    for i in range(top_elements):
        print(i+1, 'top value:', series.name, top_indexs[i])
        display(df[df[series.name] == top_indexs[i]][:repeat+1])
        print('\n')
    
def g_value_counts_perc_str(series):
    """Generate a string with resume of value counts with percentage
    Like: '[RECORRENTE(56.63%), POS-PAID(42.81%), PRE-PAID(0.55%), NENHUMA(0.00%)]'
    """
    series_name = series.name
    val_counts = series.value_counts()
    val_counts.name = 'count'
    val_percentage = series.value_counts(
        normalize=True).apply(lambda x: '{:.2%}'.format(x))
    val_percentage.name = "percentage"
    val_concat = pd.concat([val_counts, val_percentage], axis=1)
    val_concat.reset_index(level=0, inplace=True)
    val_concat = val_concat.rename(columns={'index': series_name})
    astr = '['
    for index, row in val_concat.iterrows():
        astr += str(row[series_name]) + ' (' + str(row['percentage']) + '), '
    astr = astr[:-2]
    astr += ']'
    return astr