import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

class RNN:
    def __init__(self, dir, rnn_layer_num=3):
        self.dir = Path(dir)
        inputs_1 = keras.layers.Input(shape=(9, 8))
        inputs_2 = keras.layers.Input(shape=(9,))

        embedding = keras.layers.Embedding(input_dim=60, output_dim=5)(inputs_2)
        x = keras.layers.concatenate([inputs_1, embedding], axis=-1)

        for i in range(rnn_layer_num):
            if i == rnn_layer_num - 1:
                x = keras.layers.GRU(100)(x)
            else:
                x = keras.layers.GRU(200, return_sequences=True)(x)
        outputs1 = keras.layers.Dense(5)(x)
        outputs2 = keras.layers.Dense(1)(x)
        self.model = keras.models.Model(inputs=[inputs_1, inputs_2], outputs=[outputs1, outputs2])

    def train(self, train, val, lr=0.01, epochs=1000):
        log_dir = self.dir / "log"
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(
            optimizer=optimizer,
            loss=["mean_squared_error", "mean_squared_error"],
            metrics=[None, "mean_absolute_error"],
            loss_weights=[1.0, 2.0]
        )
        self.model.fit(train, epochs=epochs,
                  validation_data=val,
                  callbacks=[keras.callbacks.EarlyStopping(monitor="val_dense_1_mean_absolute_error",
                                                           patience=5),
                             keras.callbacks.TensorBoard(log_dir)])

    def save(self):
        model_dir = self.dir / "model"
        model_dir = model_dir / "model.h5"
        self.model.save(model_dir)

    def load_model(self):
        model_dir = self.dir / "model" / "model.h5"
        self.model = keras.models.load_model(model_dir)

    def preprocess_test(self, sequence):
        x1 = sequence[:, :-1]
        x2 = sequence[:, -1]

        x1_num = x1[:, :6]
        x1_cat = x1[:, 6:]

        x1_num = (x1_num - np.array(self.train_mean)) / np.array(self.train_std)
        x2 -= 1

        x1 = np.concatenate([x1_num, x1_cat], axis=-1)

        return x1[np.newaxis, ...], x2[np.newaxis, ...]

    def predict(self, test_df, mean, std):
        submission_dir = self.dir / "data" / "sample_submission.csv"
        pred_dir = self.dir / "pred"

        submission = pd.read_csv(submission_dir)
        self.train_mean = mean
        self.train_std = std

        total_pred_lst = []
        for num in range(1, 61):
            x = np.array(test_df[test_df['num'] == num])  ## shape 177x9
            length = x.shape[0] - 9

            for i in range(length):
                x_sequence = x[i:i + 9].copy()
                x1, x2 = self.preprocess_test(x_sequence)
                pred = self.model.predict((x1, x2))

                if i % 6 == 0:
                    x[i + 9, 1] = pred[1][0][0]
                elif i % 3 == 0:
                    x[i + 9, 1:3] = np.concatenate([pred[0][0][1:2], pred[1][0][0:1]])
                else:
                    x[i + 9, 0:6] = np.concatenate([pred[0][0][0:1], pred[1][0][0:1], pred[0][0][1:]])

            total_pred_lst += [x[9:, 1]]

        total_pred = np.concatenate(total_pred_lst)
        self.pred = total_pred

        pred_dir.mkdir(exist_ok=True)
        submission['answer'] = total_pred.reshape((-1, 1))
        submission.to_csv(pred_dir / 'my_submission.csv', index=False)


class MiniRNN:
    def __init__(self, dir, rnn_layer_num=3, num=None):
        self.num = num
        self.dir = Path(dir)
        inputs = keras.layers.Input(shape=(9, 6))

        x = inputs

        for i in range(rnn_layer_num):
            if i == rnn_layer_num - 1:
                x = keras.layers.GRU(100)(x)
            else:
                x = keras.layers.GRU(200, return_sequences=True)(x)
        outputs1 = keras.layers.Dense(5)(x)
        outputs2 = keras.layers.Dense(1)(x)
        self.model = keras.models.Model(inputs=inputs, outputs=[outputs1, outputs2],
                                        name=str(self.num))

    def train(self, train, val, lr=0.01, epochs=1000):
        log_dir = self.dir / "log" / str(self.num)
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(
            optimizer=optimizer,
            loss=["mean_squared_error", "mean_squared_error"],
            metrics=[None, "mean_absolute_error"],
            loss_weights=[1.0, 2.0]
        )
        self.model.fit(train, epochs=epochs,
                  validation_data=val,
                  callbacks=[keras.callbacks.EarlyStopping(patience=5),
                             keras.callbacks.TensorBoard(log_dir)])

    def save(self):
        model_dir = self.dir / "model"
        model_dir = model_dir / "model{}.h5".format(str(self.num))
        self.model.save(model_dir)

    def load_model(self):
        model_dir = self.dir / "model" / "model{}.h5".format(str(self.num))
        self.model = keras.models.load_model(model_dir)

    def preprocess_test(self, sequence):
        x = sequence[:, :-1]

        x = x[:, :6]

        x = (x - np.array(self.train_mean)) / np.array(self.train_std)

        return x[np.newaxis, ...]

    def predict(self, test_df, mean, std):
        self.train_mean = mean
        self.train_std = std

        pred_lst = []

        x = np.array(test_df)  ## shape 177x6
        length = x.shape[0] - 9

        for i in range(length):
            x_sequence = x[i:i + 9].copy()
            x = self.preprocess_test(x_sequence)
            pred = self.model.predict(x)

            if i % 6 == 0:
                x[i + 9, 1] = pred[1][0][0]
            elif i % 3 == 0:
                x[i + 9, 1:3] = np.concatenate([pred[0][0][1:2], pred[1][0][0:1]])
            else:
                x[i + 9, 0:6] = np.concatenate([pred[0][0][0:1], pred[1][0][0:1], pred[0][0][1:]])

        self.pred = x[9:, 1].reshape((-1, 1))


class MultiRNN:
    def __init__(self, dir, rnn_layer_num=3):
        self.model_lst = [MiniRNN(dir=dir, rnn_layer_num=rnn_layer_num, num=num) for num in range(1, 61)]

    def train(self, train_lst, val_lst, lr=0.01, epochs=1000):
        for model, train, val in zip(self.model_lst, train_lst, val_lst):
            model.train(train, val, lr=lr, epochs=epochs)

    def save(self):
        for model in self.model_lst:
            model.save()

    def load_model(self):
        for model in self.model_lst:
            model.save()

    def predict(self, test_lst, mean_lst, std_lst):
        submission_dir = self.dir / "data" / "sample_submission.csv"
        pred_dir = self.dir / "pred"

        submission = pd.read_csv(submission_dir)

        total_pred_lst = []
        for model, test, mean, std in zip(self.model_lst, test_lst, mean_lst, std_lst):
            total_pred_lst += [model.predict(test, mean, std)]

        total_pred = np.concatenate(total_pred_lst, axis=0)
        self.pred = total_pred

        pred_dir.mkdir(exist_ok=True)
        submission['answer'] = total_pred
        submission.to_csv(pred_dir / 'my_submission.csv', index=False)

