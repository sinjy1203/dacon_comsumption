##
import numpy as np
from pathlib import Path
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

data_dir = Path("data")
train_dir = data_dir / "train.csv"
test_dir = data_dir / "test.csv"
train = pd.read_csv(train_dir, encoding='cp949')
test = pd.read_csv(test_dir, encoding='cp949')

##
building_num = 60
total_time = len(train) // building_num

##

