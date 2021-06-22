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
        outputs = keras.layers.Dense(6)(x)
        self.model = keras.models.Model(inputs=[inputs_1, inputs_2], outputs=outputs)

    def train(self, train, val, lr=0.01, epochs=1000):
        log_dir = self.dir / "log"
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(
            optimizer=optimizer,
            loss="mean_squared_error",
            metrics="mean_absolute_error"
        )
        self.model.fit(train, epochs=epochs,
                  validation_data=val,
                  callbacks=[keras.callbacks.EarlyStopping(patience=5),
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
                    x[i + 9, 3] = pred[0][1]
                elif i % 3 == 0:
                    x[i + 9, 2:4] = pred[0][1:3]
                else:
                    x[i + 9, 0:6] = pred[0][0:6]
            total_pred_lst += [x[9:, 1]]

        total_pred = np.concatenate(total_pred_lst)
        self.pred = total_pred

        pred_dir.mkdir(exist_ok=True)
        submission['answer'] = total_pred.reshape((-1, 1))
        submission.to_csv(pred_dir / 'my_submission.csv', index=False)
