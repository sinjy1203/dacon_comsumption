##
import argparse
from pathlib import Path

import pandas as pd

from dataset import *
from model import *

parser = argparse.ArgumentParser(description="train model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--dir", default=".", type=str, dest="dir")
parser.add_argument("--layer_num", default=3, type=int, dest="layer_num")
parser.add_argument("--lr", default=0.01, type=float, dest="lr")
parser.add_argument("--epochs", default=1000, type=int, dest="epochs")
parser.add_argument("--port", type=int, dest='port')
parser.add_argument("--mode", type=str, dest='mode')

args = parser.parse_args()

# ##
# dataset = Dataset(dir=Path(args.dir))
#
# model = RNN(dir=Path(args.dir), rnn_layer_num=args.layer_num)
#
# ##
# model.train(train=dataset.train_dataset(),
#             val=dataset.val_dataset(),
#             lr=args.lr, epochs=args.epochs)
#
# ##
# model.save()
#
# ##
# model.load_model()
#
# ##
# model.predict(dataset.test_df(), dataset.train_mean, dataset.train_std)

## multi training
multi_dataset = MultiDataset(dir=Path(args.dir))

multi_model = MultiRNN(dir=Path(args.dir), rnn_layer_num=args.layer_num)

##
multi_model.train(train_lst=multi_dataset.train_dataset(),
                  val_lst=multi_dataset.val_dataset(),
                  lr=args.lr, epochs=args.epochs)

##
multi_model.save()

##
multi_model.load_model()

##
multi_model.predict(multi_dataset.test_df(), multi_dataset.train_mean_lst,
                    multi_dataset.train_std_lst)

