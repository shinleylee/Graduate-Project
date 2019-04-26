import warnings
import itertools
import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import acf,pacf,plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARMA,ARIMA
import seaborn as sns
import random

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
    # x_aux_month = []
    # x_aux_time = []
    # x_aux_lalo = []
    # x_aux_len = []
    # x_aux_stat = []
    y = []

    for item in dataset:
        # get the length of this item
        maxWind_seq = item[7]
        item_seq_len = len(maxWind_seq)

        for i in range(1, item_seq_len):
            # month
            # aux_date = item[1][i]
            # aux_month = (aux_date//100)%100
            # x_aux_month_tensor = [aux_month-1]
            # x_aux_month.append(x_aux_month_tensor)
            # time
            # aux_time = item[2][i]
            # if aux_time == 0:
            #     x_aux_time_tensor = [1]
            # elif aux_time == 600:
            #     x_aux_time_tensor = [2]
            # elif aux_time == 1200:
            #     x_aux_time_tensor = [3]
            # elif aux_time == 1800:
            #     x_aux_time_tensor = [4]
            # else:
            #     x_aux_time_tensor = [0]
            # x_aux_time.append(x_aux_time_tensor)
            # latitude and longitude
            # x_aux_lalo_tensor = []
            # aux_latitude = item[5][i-1]
            # x_aux_lalo_tensor.append(aux_latitude)
            # aux_longitude = item[6][i-1]
            # x_aux_lalo_tensor.append(aux_longitude)
            # x_aux_lalo.append(x_aux_lalo_tensor)
            # maxWind
            x.append(maxWind_seq[:i])
            y.append(maxWind_seq[i])

    # aux_stat(statistics,including average) and aux_len
    # for item in x:
    #     x_aux_len.append(len(item))
    #     # x_aux_len.append([1-math.log(len(item))])  # length with standarizatrion
    #     sum = 0
    #     for i in item:
    #         sum = sum+i
    #     x_aux_stat.append([sum/len(item)])

    # padding: fill 0s in x to reach length of 120 (which is MAX_MAXWIND_SEQ_LEN) for lstm
    # for item in x:
    #     while len(item) < MAX_MAXWIND_SEQ_LEN:
    #         item.append(0)

    return x, y


x_train, y_train = data2tensor(train_data, MAX_MAXWIND_SEQ_LEN)
x_test, y_test = data2tensor(test_data, MAX_MAXWIND_SEQ_LEN)
# assert len(x_train)==len(x_aux_month_train) \
#     and len(x_train)==len(x_aux_time_train) \
#     and len(x_train)==len(x_aux_lalo_train) \
#     and len(x_train)==len(x_aux_len_train) \
#     and len(x_train)==len(x_aux_stat_train) \
#     and len(x_train)==len(y_train)
# assert len(x_test)==len(x_aux_month_test) \
#     and len(x_test)==len(x_aux_time_test) \
#     and len(x_test) == len(x_aux_lalo_test) \
#     and len(x_test) == len(x_aux_len_test) \
#     and len(x_test) == len(x_aux_stat_test) \
#     and len(x_test)==len(y_test)

print('The train samples are' + str(len(y_train)) + '.')
print('The test samples are ' + str(len(y_test)) + '.')
print('----------------------------------------------------------------------------------------------------')

# x_train = np.array(x_train)
# x_train_rev = x_train.tolist()
# for item in x_train_rev:
#     item.reverse()
# x_train_rev = np.array(x_train_rev)
# x_train_lstm = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
# x_aux_month_train = np.array(x_aux_month_train)
# x_aux_time_train = np.array(x_aux_time_train)
# x_aux_lalo_train = np.array(x_aux_lalo_train)
# x_aux_len_train = np.array(x_aux_len_train)
# x_aux_stat_train = np.array(x_aux_stat_train)
# y_train = np.array(y_train)

# x_test = np.array(x_test)
# x_test_rev = x_test.tolist()
# for item in x_test_rev:
#     item.reverse()
# x_test_rev = np.array(x_test_rev)
# x_test_lstm = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
# x_aux_month_test = np.array(x_aux_month_test)
# x_aux_time_test = np.array(x_aux_time_test)
# x_aux_lalo_test = np.array(x_aux_lalo_test)
# x_aux_len_test = np.array(x_aux_len_test)
# x_aux_stat_test = np.array(x_aux_stat_test)
# y_test = np.array(y_test)

# get rid of NOISE
# x_train[x_train == 67] = 65
# x_train[x_train == 77] = 75
# x_train[x_train == 84] = 85
# x_train[x_train == 93] = 95
# y_train[y_train == 67] = 65
# y_train[y_train == 77] = 75
# y_train[y_train == 84] = 85
# y_train[y_train == 93] = 95
# x_test[x_test == 67] = 65
# x_test[x_test == 77] = 75
# x_test[x_test == 84] = 85
# x_test[x_test == 93] = 95
# y_test[y_test == 67] = 65
# y_test[y_test == 77] = 75
# y_test[y_test == 84] = 85
# y_test[y_test == 93] = 95

#  print info
# print('MAX_MAXWIND_SEQ_LEN = ', MAX_MAXWIND_SEQ_LEN)
# max_maxWind = np.max(x_train)
# if np.max(x_test) > max_maxWind:
#     max_maxWind = np.max(x_test)
# max_maxWind = max_maxWind + 5
# print('Max maxWind = ', max_maxWind)
# print('----------------------------------------------------------------------------------------------------')

distribute_m = np.zeros((36,36),dtype='int32')
for idx in range(0,len(x_train)):
    i = int((x_train[idx][-1] - 10)/5)
    j = int((y_train[idx]-10)/5)
    distribute_m[i][j] = distribute_m[i][j]+1

counts_list = []
for i in range(0,36):
    sum=0
    for j in range(0,36):
        sum = sum + distribute_m[i][j]
    counts_list.append(sum)

# # draw the heatmap
# f, ax = plt.subplots(figsize=(16, 16))
# sns.heatmap(distribute_m, annot=False, ax=ax, vmax=100, vmin=0)
# plt.show()

# make predictions
prediction = []
for idx in range(0,len(x_test)):
    last_value = x_test[idx][-1]
    idx_of_last_value = int((last_value - 10) / 5)
    counts_of_last_value = counts_list[idx_of_last_value]

    if counts_of_last_value==0:  # if the predicted value never appears before
        prediction.append(last_value)
    else:
        r = random.randint(1, counts_of_last_value)
        current_counts = 0
        for i in range(0,36):
            current_counts = current_counts+distribute_m[idx_of_last_value][i]
            if current_counts>=r:
                prediction.append(i*5+10)
                break
assert len(y_test)==len(prediction)
print('Ground Truth & Prediction:')
print([int(i) for i in y_test])
print(prediction)

# calculate RMSE
score = 0
n = len(y_test)
for i in range(0,n):
    score = score + np.square(prediction[i]-y_test[i])
score = np.sqrt(score/n)
print('Calculate ',n,' sequences.')
print('RMSE=',score)
