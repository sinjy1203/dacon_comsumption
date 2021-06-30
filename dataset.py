import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, dir):
        self.data_dir = Path(dir) / "data"

        self.train_dir = self.data_dir / "train.csv"
        self.test_dir = self.data_dir / "test.csv"

        train_val = pd.read_csv(self.train_dir, encoding='cp949')
        train_val.pop('date_time')
        self.test = pd.read_csv(self.test_dir, encoding='cp949')

        self.building_num = 60
        self.total_time = len(train_val) // self.building_num
        self.val_slice = int(0.8 * self.total_time)

        train_val_num = [train_val[train_val['num'] == num] for num in list(set(train_val['num']))]
        train = pd.concat([df.iloc[:self.val_slice] for df in train_val_num])
        val = pd.concat([df.iloc[self.val_slice:] for df in train_val_num])

        self.test_columns = ['습도(%)', '전력사용량', '강수량(mm, 6시간)', '일조(hr, 3시간)',
                   '풍속(m/s)', '기온(°C)', '비전기냉방설비운영', '태양광보유', 'num']
        self.cat_columns = ['비전기냉방설비운영', '태양광보유']
        self.embedding_column = ['num']
        self.non_cat_columns = ['습도(%)', '전력사용량(kWh)', '강수량(mm)', '일조(hr)', '풍속(m/s)', '기온(°C)']

        non_cat_train = train[self.non_cat_columns]
        non_cat_val = val[self.non_cat_columns]

        self.train_mean = non_cat_train.mean()
        self.train_std = non_cat_train.std()

        self.train = pd.concat([non_cat_train, train[self.cat_columns + self.embedding_column]], axis=1)
        self.val = pd.concat([non_cat_val, val[self.cat_columns + self.embedding_column]], axis=1)

    def preprocess(self, batch):
        y1_batch = tf.concat([batch[:, -1, 0:1], batch[:, -1, 2:6]], axis=-1)
        y2_batch = batch[:, -1, 1:2]
        x1_batch_ = batch[:, :9, :-1]
        x2_batch = batch[:, :9, -1]  ## embedding

        x1_batch_num = x1_batch_[:, :, :6]
        x1_batch_cat = x1_batch_[:, :, 6:]

        x1_batch_num = (x1_batch_num - self.train_mean) / self.train_std
        x2_batch -= 1

        x1_batch = tf.concat([x1_batch_num, x1_batch_cat], axis=-1)

        x1_batch.set_shape([None, 9, 8])
        x2_batch.set_shape([None, 9])
        y1_batch.set_shape([None, 5])
        y2_batch.set_shape([None, 1])

        return (x1_batch, x2_batch), (y1_batch, y2_batch)

    def make_dataset(self, df):
        dataset = keras.preprocessing.timeseries_dataset_from_array(
            data=np.array(df, dtype=np.float32),
            targets=None,
            sequence_length=10,
            sequence_stride=10,  # 시작 데이터 사이의 간격
            sampling_rate=1,  # 한 윈도우안에서 데이터 사이의 간격
            batch_size=32,
            shuffle=True,
        )
        dataset = dataset.map(self.preprocess)
        dataset = dataset.prefetch(1)

        return dataset

    def train_dataset(self):
        return self.make_dataset(self.train)

    def val_dataset(self):
        return self.make_dataset(self.val)

    def test_df(self):
        test = self.test.copy()
        val_df = self.val.copy()
        test['전력사용량'] = np.nan
        test_df = test[self.test_columns]
        val_df.columns = self.test_columns

        new_lst = []
        for num in range(1, 61):
            train_num = val_df[val_df['num'] == num].iloc[-9:]
            test_num = test_df[test_df['num'] == num]

            test_num['비전기냉방설비운영'].iloc[:] = train_num['비전기냉방설비운영'].iloc[0]
            test_num['태양광보유'].iloc[:] = train_num['태양광보유'].iloc[0]

            new_num = pd.concat([train_num, test_num], axis=0)
            new_lst += [new_num]

        new_test_df = pd.concat(new_lst, axis=0)
        return new_test_df


class MultiDataset:
    def __init__(self, dir):
        self.data_dir = Path(dir) / "data"

        self.train_dir = self.data_dir / "train.csv"
        self.test_dir = self.data_dir / "test.csv"

        self.test_columns = ['습도(%)', '전력사용량', '강수량(mm, 6시간)', '일조(hr, 3시간)',
                             '풍속(m/s)', '기온(°C)', 'num']
        self.cat_columns = ['비전기냉방설비운영', '태양광보유']
        self.embedding_column = ['num']
        self.non_cat_columns = ['습도(%)', '전력사용량(kWh)', '강수량(mm)', '일조(hr)', '풍속(m/s)', '기온(°C)']

        train_val = pd.read_csv(self.train_dir, encoding='cp949')
        train_val.pop('date_time')
        self.test = pd.read_csv(self.test_dir, encoding='cp949')
        self.test = self.test.drop(self.cat_columns, axis=1)

        self.building_num = 60
        self.total_time = len(train_val) // self.building_num
        self.val_slice = int(0.8 * self.total_time)

        train_val_num = [train_val[train_val['num'] == num] for num in list(set(train_val['num']))]

        self.train_mean_lst = []
        self.train_std_lst = []

        self.train_lst = []
        self.val_lst = []

        for train_val in train_val_num:
            train = train_val.iloc[:self.val_slice]
            val = train_val.iloc[self.val_slice:]

            non_cat_train = train[self.non_cat_columns]
            non_cat_val = val[self.non_cat_columns]

            self.train_mean_lst += [non_cat_train.mean()]
            self.train_std_lst += [non_cat_train.std()]

            self.train_lst += [non_cat_train]
            self.val_lst += [non_cat_val]

    def preprocess(self, batch):
        y1_batch = tf.concat([batch[:, -1, 0:1], batch[:, -1, 2:6]], axis=-1)
        y2_batch = batch[:, -1, 1:2]
        x_batch = batch[:, :9, :]

        x_batch_normal = (x_batch - self.num_mean) / self.num_std

        x_batch_normal.set_shape([None, 9, 6])
        y1_batch.set_shape([None, 5])
        y2_batch.set_shape([None, 1])

        return x_batch_normal, (y1_batch, y2_batch)

    def make_dataset(self, df):
        dataset = keras.preprocessing.timeseries_dataset_from_array(
            data=np.array(df, dtype=np.float32),
            targets=None,
            sequence_length=10,
            sequence_stride=1,  # 시작 데이터 사이의 간격
            sampling_rate=1,  # 한 윈도우안에서 데이터 사이의 간격
            batch_size=32,
            shuffle=True,
        )
        dataset = dataset.map(self.preprocess)
        dataset = dataset.prefetch(1)

        return dataset

    def train_dataset(self):
        dataset_lst = []
        for train, mean, std in zip(self.train_lst, self.train_mean_lst, self.train_std_lst):
            self.num_mean = mean.to_numpy()
            self.num_std = std.to_numpy()
            dataset_lst += [self.make_dataset(train)]
        return dataset_lst

    def val_dataset(self):
        dataset_lst = []
        for val, mean, std in zip(self.val_lst, self.train_mean_lst, self.train_std_lst):
            self.num_mean = mean
            self.num_std = std
            dataset_lst += [self.make_dataset(val)]
        return dataset_lst

    def test_df(self):
        test = self.test.copy()
        val_lst = self.val_lst[:]
        test['전력사용량'] = np.nan
        test_df = test[self.test_columns]
        # val_df.columns = self.test_columns

        test_lst = []
        for i, num in enumerate(range(1, 61)):
            val_num = val_lst[i].iloc[-9:]
            val_num.columns = self.test_columns[:-1]
            # train_num = val_df[val_df['num'] == num].iloc[-9:]
            test_num = test_df[test_df['num'] == num]

            new_num = pd.concat([val_num, test_num.drop(['num'], axis=1)], axis=0)
            test_lst += [new_num]

        return test_lst
