import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as K
from keras.layers import Input, Masking, Embedding, Flatten, Dense, LSTM, Concatenate
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras import losses
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

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
    y = []

    for item in dataset:
        # get the length of this item
        maxWind_seq = item[7]
        item_seq_len = len(maxWind_seq)

        for i in range(1, item_seq_len):
            # month
            aux_date = item[1][i]
            aux_month = (aux_date//100)%100
            if aux_month==1:
                x_aux_month_tensor = [1,0,0,0,0,0,0,0,0,0,0,0]
            elif aux_month==2:
                x_aux_month_tensor = [0,1,0,0,0,0,0,0,0,0,0,0]
            elif aux_month==3:
                x_aux_month_tensor = [0,0,1,0,0,0,0,0,0,0,0,0]
            elif aux_month==4:
                x_aux_month_tensor = [0,0,0,1,0,0,0,0,0,0,0,0]
            elif aux_month==5:
                x_aux_month_tensor = [0,0,0,0,1,0,0,0,0,0,0,0]
            elif aux_month==6:
                x_aux_month_tensor = [0,0,0,0,0,1,0,0,0,0,0,0]
            elif aux_month==7:
                x_aux_month_tensor = [0,0,0,0,0,0,1,0,0,0,0,0]
            elif aux_month==8:
                x_aux_month_tensor = [0,0,0,0,0,0,0,1,0,0,0,0]
            elif aux_month==9:
                x_aux_month_tensor = [0,0,0,0,0,0,0,0,1,0,0,0]
            elif aux_month==10:
                x_aux_month_tensor = [0,0,0,0,0,0,0,0,0,1,0,0]
            elif aux_month==11:
                x_aux_month_tensor = [0,0,0,0,0,0,0,0,0,0,1,0]
            elif aux_month==12:
                x_aux_month_tensor = [0,0,0,0,0,0,0,0,0,0,0,1]
            else:
                x_aux_month_tensor = [0.0834,0.0834,0.0834,0.0834,0.0834,0.0834,0.0834,0.0834,0.0834,0.0834,0.0834,0.0834]
            x_aux_month.append(x_aux_month_tensor)
            # time
            aux_time = item[2][i]
            if aux_time == 0:
                x_aux_time_tensor = [1,0,0,0]
            elif aux_time == 600:
                x_aux_time_tensor = [0,1,0,0]
            elif aux_time == 1200:
                x_aux_time_tensor = [0,0,1,0]
            elif aux_time == 1800:
                x_aux_time_tensor = [0,0,0,1]
            else:
                x_aux_time_tensor = [0.25, 0.25, 0.25, 0.25]
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

    # fill 0s in x to reach length of 120 (which is MAX_MAXWIND_SEQ_LEN) for lstm
    for item in x:
        while len(item) < MAX_MAXWIND_SEQ_LEN:
            item.insert(0,0)

    return x, x_aux_month, x_aux_time, x_aux_lalo, y


x_train, x_aux_month_train, x_aux_time_train, x_aux_lalo_train, y_train = data2tensor(train_data, MAX_MAXWIND_SEQ_LEN)
x_test, x_aux_month_test, x_aux_time_test, x_aux_lalo_test, y_test = data2tensor(test_data, MAX_MAXWIND_SEQ_LEN)
assert len(x_train)==len(x_aux_month_train) and len(x_train)==len(x_aux_time_train) \
       and len(x_train)==len(x_aux_lalo_train) and len(x_train)==len(y_train)
assert len(x_test)==len(x_aux_month_test) and len(x_test)==len(x_aux_time_test) \
       and len(x_test) == len(x_aux_lalo_test) and len(x_test)==len(y_test)

print('The train samples are ' + str(len(y_train)) + '.')
print('The test samples are ' + str(len(y_test)) + '.')
print('----------------------------------------------------------------------------------------------------')

x_train = np.array(x_train)
x_aux_month_train = np.array(x_aux_month_train)
x_aux_time_train = np.array(x_aux_time_train)
x_aux_lalo_train = np.array(x_aux_lalo_train)

y_train = np.array(y_train)

x_test = np.array(x_test)
x_aux_month_test = np.array(x_aux_month_test)
x_aux_time_test = np.array(x_aux_time_test)
x_aux_lalo_test = np.array(x_aux_lalo_test)

y_test = np.array(y_test)
# y_test_new = []
# for item in y_test:
#     y_test_new.append(float(float(item)/200))  # normalization
# y_test = y_test_new

# normalization
max_maxWind = np.max(x_train)
if np.max(x_test) > max_maxWind:
    max_maxWind = np.max(x_test)
# x_train = x_train/max_maxWind
# y_train = y_train/max_maxWind
# x_test = x_test/max_maxWind
# y_test = y_test/max_maxWind

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

# x_train = np.concatenate([x_train, x_aux_month_train, x_aux_time_train, x_aux_lalo_train], axis=1)
# x_test = np.concatenate([x_test, x_aux_month_test, x_aux_time_test, x_aux_lalo_test], axis=1)

# min_max_scaler = MinMaxScaler(feature_range=(0,1))  # normalization
# x = np.vstack((x_train, x_test))
# x = min_max_scaler.fit_transform(x)
# split_line = x_aux_feat_train.shape[0]
# x_aux_feat_train = x_aux_feat_all[:split_line][:]
# x_aux_feat_test = x_aux_feat_all[split_line:][:]

## Prepare the data-------------------------------------------------------------------------------------------------

lr = linear_model.LinearRegression()
lr.fit(x_train, y_train)

# make predictions
prediction = lr.predict(x_test)
prediction = np.reshape(prediction,(prediction.shape[0]))
print(y_test)
print(prediction)

# calculate RMSE
score = 0
n = len(y_test)
for i in range(0,n):
    score = score + np.square(prediction[i]-y_test[i])
score = np.sqrt(score/n)
print('Calculate ',n,' sequences.')
print('RMSE=',score)
