import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as K
from keras.layers import Input, Masking, Embedding, Dense, LSTM
from keras.utils import to_categorical
from keras.models import Sequential
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


# process the data into tensors (embedding)
def data2tensor(dataset, MAX_MAXWIND_SEQ_LEN):
    x_aux = []
    x = []
    y = []
    for item in dataset:
        # aux info
        x_aux_tensor = []
        aux_time = item[2][-1]
        if aux_time == 1800:
            x_aux_tensor = [1, 0, 0, 0]
        elif aux_time == 0:
            x_aux_tensor = [0, 1, 0, 0]
        elif aux_time == 600:
            x_aux_tensor = [0, 0, 1, 0]
        elif aux_time == 1200:
            x_aux_tensor = [0, 0, 0, 1]
        else:
            aux_time = item[2][-2]
            if aux_time == 1800:
                x_aux_tensor = [1, 0, 0, 0]
            elif aux_time == 0:
                x_aux_tensor = [0, 1, 0, 0]
            elif aux_time == 600:
                x_aux_tensor = [0, 0, 1, 0]
            elif aux_time == 1200:
                x_aux_tensor = [0, 0, 0, 1]
            else:
                print('Time Error: time1 =', item[2][-1], 'time2=', item[2][-2])
                x_aux_tensor = [0.25, 0.25, 0.25, 0.25]
        aux_latitude = item[5][-1]
        x_aux_tensor.append(aux_latitude)
        aux_longitude = item[6][-1]
        x_aux_tensor.append(aux_longitude)
        # maxWind
        maxWind_seq = item[7]
        maxWind_seq_len = len(maxWind_seq)
        for i in range(1, maxWind_seq_len):
            x.append(maxWind_seq[:i])
            x_aux.append(x_aux_tensor)
            y.append(maxWind_seq[i])
    for item in x:
        while len(item) < MAX_MAXWIND_SEQ_LEN:
            item.append(0)
    return x, x_aux, y


x_train, x_aux_train, y_train = data2tensor(train_data, MAX_MAXWIND_SEQ_LEN)
x_test, x_aux_test, y_test = data2tensor(test_data, MAX_MAXWIND_SEQ_LEN)
assert len(x_train)==len(x_aux_train) and len(x_train)==len(y_train)
assert len(x_test)==len(x_aux_test) and len(x_test)==len(y_test)

print('The train samples are ' + str(len(x_train)) + '.')
print('The test samples are ' + str(len(y_test)) + '.')
print('----------------------------------------------------------------------------------------------------')

x_train = np.array(x_train)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_train = x_train/200  # normalization

y_train = np.array(y_train)
y_train = y_train/200  # normalization

x_test = np.array(x_test)
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
x_test = x_test/200  # normalization

y_test_new = []
for item in y_test:
    y_test_new.append(float(float(item)/200))  # normalization
y_test = y_test_new

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
    model = Sequential()
    #输入数据的shape为(n_samples, timestamps, features)
    #下面还有个lstm，故return_sequences设置为True
    # model.add(Embedding(input_dim=200, output_dim=EMBEDDING_DIM, input_length=MAX_MAXWIND_SEQ_LEN))
    model.add(Masking(mask_value=0, input_shape=(MAX_MAXWIND_SEQ_LEN, 1)))
    model.add(LSTM(units=1, return_sequences=False, activation='linear'))
    # model.add(LSTM(units=1, activation='linear'))
    # model.add(Dense(units=1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


model = create_model()
model.fit(x_train, y_train, batch_size=512, epochs=100, validation_split=0.1, verbose=2)

# make predictions
testPredict = model.predict(x_test)
testPredict = np.reshape(testPredict,(testPredict.shape[0]))
print(y_test)
print(testPredict)

testScore = (mean_squared_error(y_test, testPredict)) ** 0.5
testScore = testScore * 200
print('Test Score:')
print(testScore)
