##
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'NanumGothic'
pd.set_option('display.max_row', 10)
pd.set_option('display.max_columns', 10)

log_dir = Path("log")
log_dir.mkdir(exist_ok=True)

model_dir = Path('model')
model_dir.mkdir(exist_ok=True)

data_dir = Path("data")
train_dir = data_dir / "train.csv"
test_dir = data_dir / "test.csv"
train_val = pd.read_csv(train_dir, encoding='cp949')
test = pd.read_csv(test_dir, encoding='cp949')

##
date_time = pd.to_datetime(train_val.pop('date_time'), format="%Y-%m-%d %H")
num_date_time = pd.concat([train_val['num'], date_time], axis=1)

## graph
plot_columns = list(set(train_val.columns) - {'num'})
plot_features = train_val[train_val['num'] == 1][plot_columns]
plot_features.index = num_date_time[num_date_time['num'] == 1]['date_time']
plot_features.plot(subplots=True, figsize=(15, 15))

##
# print(train_val[plot_columns].describe())

##
building_num = 60
total_time = len(train_val) // building_num
val_slice = int(0.8 * total_time)

## train valid split
train_val_num = [train_val[train_val['num'] == num] for num in list(set(train_val['num']))]
train = pd.concat([df.iloc[:val_slice] for df in train_val_num])
val = pd.concat([df.iloc[val_slice:] for df in train_val_num])

## data normalization
cat_columns = ['비전기냉방설비운영', '태양광보유']
embedding_column = ['num']
label_column = ['전력사용량(kWh)']
non_cat_columns = list(set(train.columns) - set(cat_columns + embedding_column))

non_cat_train = train[non_cat_columns]
non_cat_val = val[non_cat_columns]

train_mean = non_cat_train.mean()
train_std = non_cat_train.std()

# non_cat_train = (non_cat_train - train_mean) / train_std
# non_cat_val = (non_cat_val - train_mean) / train_std

## boxplot
non_cat_train_val = train_val.drop(['num'], axis=1)[non_cat_columns]
keys_ = non_cat_train_val.keys()
non_cat_train_val = (non_cat_train_val - train_mean) / train_std
non_cat_train_val_ = non_cat_train_val.melt(var_name='Column', value_name='Normalized')

plt.figure(figsize=(15, 7))
ax = sns.violinplot(x='Column', y='Normalized', data=non_cat_train_val_)
_ = ax.set_xticklabels(keys_, rotation=90)

##
train = pd.concat([non_cat_train, train[cat_columns + embedding_column]], axis=1)
val = pd.concat([non_cat_val, val[cat_columns + embedding_column]], axis=1)

##
def preprocess(batch):
    y_batch = batch[:, 9:, :-1]
    x1_batch_ = batch[:, :9, :-1]
    x2_batch = batch[:, :9, -1]  ## embedding

    x1_batch_num = x1_batch_[:, :, :6]
    x1_batch_cat = x1_batch_[:, :, 6:]

    x1_batch_num = (x1_batch_num - train_mean) / train_std
    x2_batch -= 1

    x1_batch = tf.concat([x1_batch_num, x1_batch_cat], axis=-1)

    x1_batch.set_shape([None, 9, 8])
    x2_batch.set_shape([None, 9])
    y_batch.set_shape([None, 1, 8])

    return (x1_batch, x2_batch), (y_batch[:, 0, :6], y_batch[:, 0, 6], y_batch[:, 0, 7])

def preprocess_test(sequence): ## sequence shape: 9x9
    x1 = sequence[:, :-1]
    x2 = sequence[:, -1]

    x1_num = x1[:, :6]
    x1_cat = x1[:, 6:]

    x1_num = (x1_num - np.array(train_mean)) / np.array(train_std)
    x2 -= 1

    x1 = np.concatenate([x1_num, x1_cat], axis=-1)

    return x1[np.newaxis, ...], x2[np.newaxis, ...]

def make_dataset(df):
    dataset = keras.preprocessing.timeseries_dataset_from_array(
        data=np.array(df, dtype=np.float32),
        targets=None,
        sequence_length=10,
        sequence_stride=10, # 시작 데이터 사이의 간격
        sampling_rate=1, # 한 윈도우안에서 데이터 사이의 간격
        batch_size=32,
        shuffle=True,
    )
    dataset = dataset.map(preprocess)
    dataset = dataset.prefetch(1)

    return dataset

##
train_dataset = make_dataset(train)
val_dataset = make_dataset(val)

##
rnn_layer_num = 3
inputs_1 = keras.layers.Input(shape=(9, 8))
inputs_2 = keras.layers.Input(shape=(9,))

embedding = keras.layers.Embedding(input_dim=60, output_dim=5)(inputs_2)
x = keras.layers.concatenate([inputs_1, embedding], axis=-1)

for i in range(rnn_layer_num):
    if i == rnn_layer_num - 1:
        x = keras.layers.GRU(100)(x)
    else:
        x = keras.layers.GRU(200, return_sequences=True)(x)
outputs_1 = keras.layers.Dense(6, name='num_output')(x)
outputs_2 = keras.layers.Dense(1, activation='sigmoid', name='cold_output')(x)
outputs_3 = keras.layers.Dense(1, activation='sigmoid', name='sun_output')(x)
model = keras.models.Model(inputs=[inputs_1, inputs_2],
                           outputs=[outputs_1, outputs_2, outputs_3])

## plot model
plot_model = keras.utils.plot_model(model, "model", show_shapes=True)

##
optimizer = keras.optimizers.Adam(learning_rate=0.01)
model.compile(
    optimizer=optimizer,
    loss={
        'num_output': keras.losses.MeanSquaredError(),
        'cold_output': keras.losses.BinaryCrossentropy(),
        'sun_output': keras.losses.BinaryCrossentropy()
    },
    loss_weights=[1.0, 1.0, 1.0],
    metrics={
        'num_output': keras.metrics.MeanAbsoluteError(),
        'cold_output': keras.metrics.Accuracy(),
        'sun_output': keras.metrics.Accuracy()
    }
)
model.fit(train_dataset, epochs=1000,
          validation_data=val_dataset,
          callbacks=[keras.callbacks.EarlyStopping(patience=5),
                     keras.callbacks.TensorBoard(log_dir)])

##
model.save(model_dir / "GRU3.h5")

##
model = keras.models.load_model(model_dir / "GRU3.h5")

## 예측하기
submission_dir = data_dir / "sample_submission.csv"
submission = pd.read_csv(submission_dir)

##
def prepare_test(test_df, train_df):
    columns = ['일조(hr, 3시간)', '습도(%)', '강수량(mm, 6시간)', '전력사용량',
               '풍속(m/s)', '기온(°C)', '비전기냉방설비운영', '태양광보유', 'num']
    test_df['전력사용량'] = np.nan
    test_df_new = test_df[columns]
    train_df.columns = columns

    new_lst = []

    for num in range(1, 61):
        train_num = train_df[train_df['num'] == num].iloc[-9:]
        test_num = test_df_new[test_df_new['num'] == num]

        test_num['비전기냉방설비운영'] = train_num['비전기냉방설비운영'].iloc[0]
        test_num['태양광보유'] = train_num['태양광보유'].iloc[0]

        new_num = pd.concat([train_num, test_num], axis=0)
        new_lst += [new_num]

    new_df = pd.concat(new_lst, axis=0)
    return new_df

##
test_df = prepare_test(test, val)

##
total_pred_lst = []
for num in range(1, 61):
    x = np.array(test_df[test_df['num'] == num])  ## shape 177x9
    length = x.shape[0] - 9

    for i in range(length):
        x_sequence = x[i:i+9].copy()
        x1, x2 = preprocess_test(x_sequence)
        pred = model.predict((x1, x2))

        if i % 6 == 0:
            x[i+9, 3] = pred[0][0][3]
        elif i % 3 == 0:
            x[i+9, 2:4] = pred[0][0][2:4]
        else:
            x[i+9, 0:6] = pred[0][0][0:6]
    total_pred_lst += [x[9:, 3]]

##
total_pred = np.concatenate(total_pred_lst)

##
pred_dir = Path('pred')
pred_dir.mkdir(exist_ok=True)
submission['answer'] = total_pred.reshape((-1, 1))
submission.to_csv(pred_dir / 'my_submission.csv', index=False)

##
a = np.arange(10)
b = a[:5]
b -= 1
