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

DATA_PATH = './dataset/'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
FIX_LEN = 19
DRAW_CURVES = True
TRAIN_LEN = 4
MAX_MAXWIND_SEQ_LEN = -1
EMBEDDING_DIM = 32

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


x_train, x_aux_month_train, x_aux_time_train, x_aux_lalo_train, x_aux_len_train, x_aux_stat_train, y_train = data2tensor(train_data, MAX_MAXWIND_SEQ_LEN)
x_test, x_aux_month_test, x_aux_time_test, x_aux_lalo_test, x_aux_len_test, x_aux_stat_test, y_test = data2tensor(test_data, MAX_MAXWIND_SEQ_LEN)
assert len(x_train)==len(x_aux_month_train) and len(x_train)==len(x_aux_time_train) \
       and len(x_train)==len(x_aux_lalo_train) and len(x_train)==len(y_train)
assert len(x_test)==len(x_aux_month_test) and len(x_test)==len(x_aux_time_test) \
       and len(x_test) == len(x_aux_lalo_test) and len(x_test)==len(y_test)

print('The train samples are ' + str(len(y_train)) + '.')
print('The test samples are ' + str(len(y_test)) + '.')
print('----------------------------------------------------------------------------------------------------')

x_train = np.array(x_train)
x_train_rev = x_train.tolist()
for item in x_train_rev:
    item.reverse()
x_train_rev = np.array(x_train_rev)
x_train_lstm = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_aux_month_train = np.array(x_aux_month_train)
x_aux_time_train = np.array(x_aux_time_train)
x_aux_lalo_train = np.array(x_aux_lalo_train)
x_aux_len_train = np.array(x_aux_len_train)
x_aux_stat_train = np.array(x_aux_stat_train)
y_train = np.array(y_train)

x_test = np.array(x_test)
x_test_rev = x_test.tolist()
for item in x_test_rev:
    item.reverse()
x_test_rev = np.array(x_test_rev)
x_test_lstm = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
x_aux_month_test = np.array(x_aux_month_test)
x_aux_time_test = np.array(x_aux_time_test)
x_aux_lalo_test = np.array(x_aux_lalo_test)
x_aux_len_test = np.array(x_aux_len_test)
x_aux_stat_test = np.array(x_aux_stat_test)
y_test = np.array(y_test)

# normalization
max_maxWind = np.max(x_train)
if np.max(x_test) > max_maxWind:
    max_maxWind = np.max(x_test)

x_train = x_train/max_maxWind
x_train_rev = x_train_rev/max_maxWind
x_train_lstm = x_train_lstm/max_maxWind
x_aux_lalo_train = x_aux_lalo_train/90
x_aux_stat_train = x_aux_stat_train/max_maxWind
y_train = y_train/max_maxWind

x_test = x_test/max_maxWind
x_test_rev = x_test_rev/max_maxWind
x_test_lstm = x_test_lstm/max_maxWind
x_aux_lalo_test = x_aux_lalo_test/90
x_aux_stat_test = x_aux_stat_test/max_maxWind
y_test = y_test/max_maxWind


# min_max_scaler = MinMaxScaler(feature_range=(0,1))  # normalization
# x_aux_feat_all = np.vstack((x_aux_feat_train, x_aux_feat_test))
# x_aux_feat_all = min_max_scaler.fit_transform(x_aux_feat_all)
# split_line = x_aux_feat_train.shape[0]
# x_aux_feat_train = x_aux_feat_all[:split_line][:]
# x_aux_feat_test = x_aux_feat_all[split_line:][:]

# # draw curves
# if DRAW_CURVES:
#     for item in test_data:
#         plt.plot(item[7])
#     plt.xlabel('time')
#     plt.xlim(0, 90)
#     plt.ylabel('max wind')
#     plt.ylim(0, 200)
#     plt.title('All curves')
#     plt.show()

## Prepare the data-------------------------------------------------------------------------------------------------

def create_model():
    #输入数据的shape为(n_samples, timestamps, features)
    main_input = Input(shape=(MAX_MAXWIND_SEQ_LEN,), name='main_input')
    main_input_rev = Input(shape=(MAX_MAXWIND_SEQ_LEN,), name='main_input_rev')
    main_input_lstm = Input(shape=(MAX_MAXWIND_SEQ_LEN,1), name='main_input_lstm')
    aux_month_input = Input(shape=(1,), name='aux_month_input')
    aux_time_input = Input(shape=(1,), name='aux_time_input')
    aux_lalo_input = Input(shape=(2,), name='aux_lalo_input')
    aux_len_input = Input(shape=(1,), name='aux_len_input')
    aux_stat_input = Input(shape=(1,), name='aux_stat_input')

    # masking = Masking(mask_value=0)(main_input_lstm)
    em_lstm = Embedding(input_dim=200, output_dim=128, input_length=MAX_MAXWIND_SEQ_LEN, mask_zero=True)(main_input)
    lstm = LSTM(4)(em_lstm)

    # att = Dense(MAX_MAXWIND_SEQ_LEN, activation='softmax', name='this_dense')(lstm)
    # a_probs = Multiply()([lstm, att])
    # a = Reshape((MAX_MAXWIND_SEQ_LEN, 1))(a_probs)

    # aux_month_info = Embedding(input_dim=13, output_dim=4, input_length=1)(aux_month_input)
    # aux_month_info = Flatten()(aux_month_info)

    # aux_time_info = Embedding(input_dim=5, output_dim=4, input_length=1)(aux_time_input)
    # aux_time_info = Flatten()(aux_time_info)

    # aux_lalo_info = Dense(4, activation='sigmoid')(aux_lalo_input)

    # aux = Concatenate()([aux_month_info, aux_time_info, aux_lalo_info])
    # aux_add = Add()([aux_month_info, aux_time_info, aux_lalo_info])
    # aux_product = Multiply()([aux_month_info, aux_time_info, aux_lalo_info])
    # aux_deep = Dense(12, activation='relu')(aux)
    # x = Concatenate()([aux_add, aux_deep, aux_product])
    # x = Dense(6, activation='relu')(x)
    # x = Dense(3, activation='relu')(a)



    # o = Concatenate()([lstm, aux_stat_input])
    main_output = Dense(1, activation='sigmoid', name='finalDense')(lstm)

    #下面还有个lstm，故return_sequences设置为True
    # model.add(Masking(mask_value=0, input_shape=(MAX_MAXWIND_SEQ_LEN, 1)))
    # model.add(LSTM(units=1, return_sequences=False, activation='linear'))
    # model.add(LSTM(units=1, activation='linear'))
    # model.add(Dense(units=1, activation='linear'))

    model = Model(inputs=[main_input, main_input_rev, main_input_lstm, aux_month_input, aux_time_input, aux_lalo_input, aux_len_input, aux_stat_input], outputs=main_output)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


model = create_model()
model.fit([x_train, x_train_rev, x_train_lstm, x_aux_month_train, x_aux_time_train, x_aux_lalo_train, x_aux_len_train, x_aux_stat_train], y_train,
          batch_size=512, epochs=100, validation_split=0.1, verbose=2)

# make predictions
testPredict = model.predict([x_test, x_test_rev, x_test_lstm, x_aux_month_test, x_aux_time_test, x_aux_lalo_test, x_aux_len_test, x_aux_stat_test])
testPredict = np.reshape(testPredict,(testPredict.shape[0]))
print(y_test)
print(testPredict)

testScore = (mean_squared_error(y_test, testPredict)) ** 0.5
testScore = testScore * max_maxWind
print('Test Score:')
print(testScore)









x_test = x_test.tolist()
num_li = []
for item in x_test:
    num = 0
    for i in item:
        if i!=0:
            num = num+1
    num_li.append(num)

substract_li = (y_test - testPredict) * max_maxWind
plt.plot(num_li, color='blue')
plt.plot(substract_li, color='red')
plt.plot(y_test*max_maxWind, color='green')
plt.plot(testPredict*max_maxWind, color='orange') # linestyle="--")
plt.xlabel('test samples')
plt.ylabel('max wind')
plt.ylim(-100, 200)
plt.title('All curves')
plt.show()
