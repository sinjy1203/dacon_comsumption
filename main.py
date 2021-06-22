##
import argparse
from pathlib import Path

import pandas as pd

from dataset import Dataset
from model import RNN

parser = argparse.ArgumentParser(description="train model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--dir", default=".", type=str, dest="dir")
parser.add_argument("--layer_num", default=3, type=int, dest="layer_num")
parser.add_argument("--lr", default=0.01, type=float, dest="lr")
parser.add_argument("--epochs", default=1000, type=int, dest="epochs")
parser.add_argument("--port", type=int, dest='port')
parser.add_argument("--mode", type=str, dest='mode')

args = parser.parse_args()

dataset = Dataset(dir=Path(args.dir))

model = RNN(dir=Path(args.dir), rnn_layer_num=args.layer_num)

##
model.train(train=dataset.train_dataset(),
            val=dataset.val_dataset(),
            lr=args.lr, epochs=args.epochs)

##
model.save()

##
model.load_model()

##
model.predict(dataset.test_df(), dataset.train_mean, dataset.train_std)
