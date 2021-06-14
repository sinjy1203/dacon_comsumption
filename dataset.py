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

    def preprocess(self):