##
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

plt.rcParams['font.family'] = 'NanumGothic'
pd.set_option('display.max_row', 10)
pd.set_option('display.max_columns', 10)

data_dir = Path("data")
train_dir = data_dir / "train.csv"
test_dir = data_dir / "test.csv"
train_val = pd.read_csv(train_dir, encoding='cp949')
test = pd.read_csv(test_dir, encoding='cp949')

##
date_time = pd.to_datetime(train_val.pop('date_time'), format="%Y-%m-%d %H")
num_date_time = pd.concat([train_val['num'], date_time], axis=1)

##
plot_columns = list(set(train_val.columns) - {'num'})
plot_features = train_val[train_val['num'] == 1][plot_columns]
plot_features.index = num_date_time[num_date_time['num'] == 1]['date_time']
plot_features.plot(subplots=True, figsize=(15, 15))

##
print(train_val[plot_columns].describe())

##
building_num = 60
total_time = len(train_val) // building_num
val_slice = int(0.8 * total_time)

## train valid split
train_val_num = [train_val[train_val['num'] == num] for num in list(set(train_val['num']))]
train = pd.concat([df.iloc[:val_slice] for df in train_val_num])
val = pd.concat([df.iloc[val_slice:] for df in train_val_num])

##
train, val = train.drop(['num'], axis=1), val.drop(['num'], axis=1)

## data normalization
cat_columns = ['비전기냉방설비운영', '태양광보유']
non_cat_columns = list(set(train.columns) - set(cat_columns))

non_cat_train = train[non_cat_columns]
non_cat_val = val[non_cat_columns]

train_mean = non_cat_train.mean()
train_std = non_cat_train.std()

non_cat_train = (non_cat_train - train_mean) / train_std
non_cat_val = (non_cat_val - train_mean) / train_std

##
non_cat_train_val = train_val.drop(['num'], axis=1)[non_cat_columns]
keys_ = non_cat_train_val.keys()
non_cat_train_val = (non_cat_train_val - train_mean) / train_std
non_cat_train_val_ = non_cat_train_val.melt(var_name='Column', value_name='Normalized')

##
plt.figure(figsize=(15, 7))
ax = sns.violinplot(x='Column', y='Normalized', data=non_cat_train_val_)
_ = ax.set_xticklabels(keys_, rotation=90)

##
train_df = pd.concat([non_cat_train, train[cat_columns]], axis=1)
val_df = pd.concat([non_cat_val, val[cat_columns]], axis=1)

##
class WindowGenerator():
    def __init__(self, input_width=):

##
train_dataset = keras.preprocessing.timeseries_dataset_from_array(
    data=np.array(train_df, dtype=np.float32)
    
)
