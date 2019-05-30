import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as K
from keras.layers import Input, Masking, Embedding, Flatten, Dense, LSTM, Concatenate, Multiply, Add, Permute, Reshape, Bidirectional, Conv1D, Activation, SpatialDropout1D, GRU, Dropout
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras import losses
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras import backend as Kbe
from keras.models import load_model

TYPHOON = 'EP012014'
DATA_PATH = './demo/'
TEST_FILE = TYPHOON+'.csv'
MODEL_PATH = './demo/'
MODEL_NAME = 'NCF(TCN)32_20.h5'
MAX_MAXWIND_SEQ_LEN = 120
DROPOUT_RATE = 0.8
IMG_SAVE_PATH = TYPHOON+'.png'

# read dataset to dataframe
print('Loading data...')
test_data_df = pd.read_csv(DATA_PATH + TEST_FILE, index_col=0)
print(test_data_df.head())
print('Data load finish.')
print('----------------------------------------------------------------------------------------------------')

# convert dataframe from string to list
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
for item in test_data:
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
            item.insert(0,0)

    return x, x_aux_month, x_aux_time, x_aux_lalo, x_aux_len, x_aux_stat, y

x_test, x_aux_month_test, x_aux_time_test, x_aux_lalo_test, x_aux_len_test, x_aux_stat_test, y_test = data2tensor(test_data,MAX_MAXWIND_SEQ_LEN)
assert len(x_test)==len(x_aux_month_test) and len(x_test)==len(x_aux_time_test) \
       and len(x_test) == len(x_aux_lalo_test) and len(x_test)==len(y_test)

print('The test samples are ' + str(len(y_test)) + '.')
print('----------------------------------------------------------------------------------------------------')


x_test = np.array(x_test, dtype='int32')
y_test = np.array(y_test, dtype='int32')

# get rid of NOISE
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

user_num_test = x_test.shape[0]
user_test = x_test.tolist()
user_test_new = []
for item in user_test:
    for i in range(0, maxWind_li_num):
        user_test_new.append(item)
user_test = np.array(user_test_new, dtype='int32')
user_test_lstm = user_test.reshape([user_test.shape[0],user_test.shape[1],1])
item_test = []
for i in range(0,user_num_test):
    for item in maxWind_li_input:
        item_test.append(item)
item_test = np.array(item_test, dtype='int32')

## Prepare the data-------------------------------------------------------------------------------------------------

model = load_model(MODEL_PATH+MODEL_NAME)

print('calculate rmse-------------------------------------------------------------------------------------------------')

# make predictions
testPredict = model.predict([user_test_lstm, item_test])
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
rmse = mse**0.5
print('RMSE=',rmse)

# draw
x, = plt.plot(y_pred, color='blue')
y, = plt.plot(y_test, color='red')#linestyle="--"

x_ticks = [i for i in range(0,len(y_pred))]  # the ticks of x axis, used for plt.bar
z = y_test - y_pred
z = plt.bar(x=x_ticks, height=z, color='gray')

xaxis = y_test-y_test  # the x axis
xaxis, = plt.plot(xaxis, color='black')

plt.xlabel('time step')
plt.ylabel('maxWind')
plt.ylim(-50, 200)
plt.legend([x,y,z],['Prediction','Ground Truth','Differences'])
plt.title('Prediction and Ground Truth Curves')
plt.savefig(MODEL_PATH+IMG_SAVE_PATH)
plt.show()
