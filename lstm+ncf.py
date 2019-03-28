import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as K
from keras.layers import Input, Masking, Embedding, Flatten, Dense, LSTM, Concatenate, Multiply, Add, Permute, Reshape
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras import losses
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras import backend as Kbe

DATA_PATH = './dataset/'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
MAX_MAXWIND_SEQ_LEN = -1

# read dataset to dataframe
print('Loading data...')
train_data_df = pd.read_csv(DATA_PATH + TRAIN_FILE, index_col=0)
print(train_data_df.head())
test_data_df = pd.read_csv(DATA_PATH + TEST_FILE, index_col=0)
print(test_data_df.head())
print('Data load finish.')
print('----------------------------------------------------------------------------------------------------')

# convert dataframe from string to list
train_data = []
for index, item in train_data_df.iterrows():
    row = [item['ID']]
    row.append(list(map(int, item['Date'][1:-1].split(','))))
    row.append(list(map(int, item['Time'][1:-1].split(','))))
    row.append(list(item['Event'][1:-1].split(',')))
    row.append(list(item['Status'][1:-1].split(',')))
    row.append(list(map(float, item['Latitude'][1:-1].split(','))))
    row.append(list(map(float, item['Longitude'][1:-1].split(','))))
    row.append(list(map(float, item['MaxWind'][1:-1].split(','))))
    train_data.append(row)

test_data = []
for index, item in test_data_df.iterrows():
    row = [item['ID']]
    row.append(list(map(int, item['Date'][1:-1].split(','))))
    row.append(list(map(int, item['Time'][1:-1].split(','))))
    row.append(list(item['Event'][1:-1].split(',')))
    row.append(list(item['Status'][1:-1].split(',')))
    row.append(list(map(float, item['Latitude'][1:-1].split(','))))
    row.append(list(map(float, item['Longitude'][1:-1].split(','))))
    row.append(list(map(float, item['MaxWind'][1:-1].split(','))))
    test_data.append(row)
# trian/test data list index:
    # 0 ID
    # 1 Date
    # 2 Time
    # 3 Event
    # 4 Status
    # 5 Latitude
    # 6 Longitude
    # 7 MaxWind

# get the MAX_SEQ_LEN
for item in train_data + test_data:
    maxLen = len(item[7])
    if maxLen - 1 > MAX_MAXWIND_SEQ_LEN:
        MAX_MAXWIND_SEQ_LEN = maxLen - 1

# process the data into tensors
def data2tensor(dataset, MAX_MAXWIND_SEQ_LEN):
    x = []
    x_aux_month = []
    x_aux_time = []
    x_aux_lalo = []
    x_aux_stat = []
    x_aux_len = []
    y = []

    for item in dataset:
        # get the length of this item
        maxWind_seq = item[7]
        item_seq_len = len(maxWind_seq)

        for i in range(1, item_seq_len):
            # month
            aux_date = item[1][i]
            aux_month = (aux_date//100)%100
            x_aux_month_tensor = [aux_month-1]
            x_aux_month.append(x_aux_month_tensor)
            # time
            aux_time = item[2][i]
            if aux_time == 0:
                x_aux_time_tensor = [1]
            elif aux_time == 600:
                x_aux_time_tensor = [2]
            elif aux_time == 1200:
                x_aux_time_tensor = [3]
            elif aux_time == 1800:
                x_aux_time_tensor = [4]
            else:
                x_aux_time_tensor = [0]
            x_aux_time.append(x_aux_time_tensor)
            # latitude and longitude
            x_aux_lalo_tensor = []
            aux_latitude = item[5][i-1]
            x_aux_lalo_tensor.append(aux_latitude)
            aux_longitude = item[6][i-1]
            x_aux_lalo_tensor.append(aux_longitude)
            x_aux_lalo.append(x_aux_lalo_tensor)
            # maxWind
            x.append(maxWind_seq[:i])
            y.append(maxWind_seq[i])

    for item in x:
        x_aux_len.append([1-math.log(len(item))])
        sum = 0
        for i in item:
            sum = sum+i
        x_aux_stat.append([sum/len(item)])

    # fill 0s in x to reach length of 120 (which is MAX_MAXWIND_SEQ_LEN) for lstm
    for item in x:
        while len(item) < MAX_MAXWIND_SEQ_LEN:
            item.append(0)

    return x, x_aux_month, x_aux_time, x_aux_lalo, x_aux_len, x_aux_stat, y

x_train, x_aux_month_train, x_aux_time_train, x_aux_lalo_train, x_aux_len_train, x_aux_stat_train, y_train = data2tensor(train_data,MAX_MAXWIND_SEQ_LEN)
x_test, x_aux_month_test, x_aux_time_test, x_aux_lalo_test, x_aux_len_test, x_aux_stat_test, y_test = data2tensor(test_data,MAX_MAXWIND_SEQ_LEN)
assert len(x_train)==len(x_aux_month_train) and len(x_train)==len(x_aux_time_train) \
       and len(x_train)==len(x_aux_lalo_train) and len(x_train)==len(y_train)
assert len(x_test)==len(x_aux_month_test) and len(x_test)==len(x_aux_time_test) \
       and len(x_test) == len(x_aux_lalo_test) and len(x_test)==len(y_test)

print('The train samples are ' + str(len(y_train)) + '.')
print('The test samples are ' + str(len(y_test)) + '.')
print('----------------------------------------------------------------------------------------------------')

x_train = np.array(x_train, dtype='int32')
y_train = np.array(y_train, dtype='int32')

x_test = np.array(x_test, dtype='int32')
y_test = np.array(y_test, dtype='int32')

# get rid of NOISE
x_train[x_train == 67] = 65
x_train[x_train == 77] = 75
x_train[x_train == 84] = 85
x_train[x_train == 93] = 95
y_train[y_train == 67] = 65
y_train[y_train == 77] = 75
y_train[y_train == 84] = 85
y_train[y_train == 93] = 95
x_test[x_test == 67] = 65
x_test[x_test == 77] = 75
x_test[x_test == 84] = 85
x_test[x_test == 93] = 95
y_test[y_test == 67] = 65
y_test[y_test == 77] = 75
y_test[y_test == 84] = 85
y_test[y_test == 93] = 95

maxWind_li_input = [[x] for x in range(10,190,5)]
maxWind_li_num = len(maxWind_li_input)

user_num_train = x_train.shape[0]
user_train_pos = x_train
item_train_pos = y_train.reshape((y_train.shape[0],1))
rate_train_pos = [1 for x in range(0,user_num_train)]

user_train_neg = x_train.tolist()
user_train_neg_new = []
for item in user_train_neg:
    for i in range(0,maxWind_li_num-1):
        user_train_neg_new.append(item)
user_train_neg = np.array(user_train_neg_new)
item_train_neg = []
for pos_num in item_train_pos:
    for neg_num in maxWind_li_input:
        if pos_num[0] not in maxWind_li_input:
            print('E: ', pos_num[0], 'not in maxWind_list.')
        if pos_num[0] != neg_num:
            item_train_neg.append(neg_num)
item_train_neg = np.array(item_train_neg, dtype='int32')
rate_train_neg = [0 for x in range(0,user_num_train*(maxWind_li_num-1))]

user_train = np.concatenate((user_train_pos, user_train_neg),axis=0)
item_train = np.concatenate((item_train_pos, item_train_neg),axis=0)
rate_train = np.array(rate_train_pos + rate_train_neg)

user_num_test = x_test.shape[0]
user_test = x_test.tolist()
user_test_new = []
for item in user_test:
    for i in range(0, maxWind_li_num):
        user_test_new.append(item)
user_test = np.array(user_test_new, dtype='int32')
item_test = []
for i in range(0,user_num_test):
    for item in maxWind_li_input:
        item_test.append(item)
item_test = np.array(item_test, dtype='int32')

## Prepare the data-------------------------------------------------------------------------------------------------


def create_model():
    #输入数据的shape为(n_samples, timestamps, features)
    user_input = Input(shape=(MAX_MAXWIND_SEQ_LEN,), dtype='int32', name='user_input')
    user_em8 = Embedding(input_dim=190, output_dim=8, input_length=MAX_MAXWIND_SEQ_LEN, mask_zero=True)(user_input)
    user_em8 = LSTM(1)(user_em8)
    user_em16 = Embedding(input_dim=190, output_dim=16, input_length=MAX_MAXWIND_SEQ_LEN, mask_zero=True)(user_input)
    user_em16 = LSTM(1)(user_em16)
    user_em32 = Embedding(input_dim=190, output_dim=32, input_length=MAX_MAXWIND_SEQ_LEN, mask_zero=True)(user_input)
    user_em32 = LSTM(1)(user_em32)
    user_em64 = Embedding(input_dim=190, output_dim=64, input_length=MAX_MAXWIND_SEQ_LEN, mask_zero=True)(user_input)
    user_em64 = LSTM(1)(user_em64)
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    item_em8 = Embedding(input_dim=190, output_dim=8,  input_length=1)(item_input)
    item_em8 = Flatten()(item_em8)
    item_em16 = Embedding(input_dim=190, output_dim=16, input_length=1)(item_input)
    item_em16 = Flatten()(item_em16)
    item_em32 = Embedding(input_dim=190, output_dim=32, input_length=1)(item_input)
    item_em32 = Flatten()(item_em32)
    item_em64 = Embedding(input_dim=190, output_dim=64, input_length=1)(item_input)
    item_em64 = Flatten()(item_em64)

    user_high = Dense(32, activation='relu')(user_em64)
    item_high = Dense(32, activation='relu')(item_em64)

    add = Add()([user_high, item_high])
    mul = Multiply()([user_high, item_high])
    concat = Concatenate()([user_high, item_high])
    deep = Dense(32, activation='relu')(concat)

    ncf = Concatenate()([add, mul, concat, deep])
    main_output = Dense(1, activation='sigmoid', name='finalDense')(ncf)

    model = Model(inputs=[user_input, item_input], outputs=main_output)
    model.compile(loss='mean_squared_error', optimizer='adam')  # binary_crossentropy
    return model

model = create_model()
model.fit([user_train, item_train], rate_train, batch_size=1024, epochs=100, validation_split=0.1, verbose=2)

print('calculate rmse-------------------------------------------------------------------------------------------------')

# make predictions
testPredict = model.predict([user_test, item_test])
testPredict = np.reshape(testPredict,(testPredict.shape[0]))
print(testPredict)

y_pred = []
for i in range(0, user_num_test):
    sigmoids = []
    maxPredictWind = 0
    for j in range(0,maxWind_li_num):
        idx = int(i*maxWind_li_num + j)
        sigmoids.append(testPredict[idx])
    assert len(sigmoids) == maxWind_li_num
    maxPredictWind = maxWind_li_input[sigmoids.index(max(sigmoids))][0]
    y_pred.append(maxPredictWind)

print('groundTruth and prediction:')
print(y_test.tolist())
print(y_pred)

mse = mean_squared_error(y_test,y_pred)
print('rmse=',mse**0.5)
