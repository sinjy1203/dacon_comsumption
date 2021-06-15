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
    def __init__(self, data_dir):
        self.train_dir = data_dir / "train.csv"
        self.test_dir = data_dir / "test.csv"

        train_val = pd.read_csv(self.train_dir, encoding='cp949')
        self.test = pd.read_csv(self.test_dir, encoding='cp949')

        self.building_num = 60
        self.total_time = len(train_val) // self.building_num
        self.val_slice = int(0.8 * self.total_time)

        train_val_num = [train_val[train_val['num'] == num] for num in list(set(train_val['num']))]
        train = pd.concat([df.iloc[:self.val_slice] for df in train_val_num])
        val = pd.concat([df.iloc[self.val_slice:] for df in train_val_num])

        self.test_columns = ['일조(hr, 3시간)', '습도(%)', '강수량(mm, 6시간)', '전력사용량',
                   '풍속(m/s)', '기온(°C)', '비전기냉방설비운영', '태양광보유', 'num']
        self.cat_columns = ['비전기냉방설비운영', '태양광보유']
        self.embedding_column = ['num']
        self.label_column = ['전력사용량(kWh)']
        non_cat_columns = list(set(train.columns) - set(self.cat_columns + self.embedding_column))

        non_cat_train = train[non_cat_columns]
        non_cat_val = val[non_cat_columns]

        self.train_mean = non_cat_train.mean()
        self.train_std = non_cat_train.std()

        self.train = pd.concat([non_cat_train, train[self.cat_columns + self.embedding_column]], axis=1)
        self.val = pd.concat([non_cat_val, val[self.cat_columns + self.embedding_column]], axis=1)

    def preprocess(self, batch):
        y_batch = batch[:, 9:, :-1]
        x1_batch_ = batch[:, :9, :-1]
        x2_batch = batch[:, :9, -1]  ## embedding

        x1_batch_num = x1_batch_[:, :, :6]
        x1_batch_cat = x1_batch_[:, :, 6:]

        x1_batch_num = (x1_batch_num - self.train_mean) / self.train_std
        x2_batch -= 1

        x1_batch = tf.concat([x1_batch_num, x1_batch_cat], axis=-1)

        x1_batch.set_shape([None, 9, 8])
        x2_batch.set_shape([None, 9])
        y_batch.set_shape([None, 1, 8])

        return (x1_batch, x2_batch), (y_batch[:, 0, :6], y_batch[:, 0, 6], y_batch[:, 0, 7])

    def preprocess_test(self, sequence):
        x1 = sequence[:, :-1]
        x2 = sequence[:, -1]

        x1_num = x1[:, :6]
        x1_cat = x1[:, 6:]

        x1_num = (x1_num - np.array(self.train_mean)) / np.array(self.train_std)
        x2 -= 1

        x1 = np.concatenate([x1_num, x1_cat], axis=-1)

        return x1[np.newaxis, ...], x2[np.newaxis, ...]

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
        test_df = (self.test)
        test_df = test_df[self.test_columns]
        self.
