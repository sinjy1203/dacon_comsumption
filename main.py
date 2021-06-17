import argparse
from pathlib import Path
from dataset import Dataset
from model import RNN

parser = argparse.ArgumentParser(description="train model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--data_dir", default="data", type=str, dest="data_dir")
parser.add_argument("--layer_num", default=3, type=int, dest="layer_num")
parser.add_argument("--lr", default=0.01, type=float, dest="lr")
parser.add_argument("--epochs", default=1000, type=int, dest="epochs")
parser.add_argument("--log_dir", default="log", type=str, dest="log_dir")

args = parser.parse_args()

dataset = Dataset(data_dir=Path(args.data_dir))
model = RNN(rnn_layer_num=args.layer_num)

model.train(train=dataset.train_dataset(),
            val=dataset.val_dataset(),
            lr=args.lr, epochs=args.epochs, log_dir=Path(args.log_dir))

