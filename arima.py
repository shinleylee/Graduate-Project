import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import acf,pacf,plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARMA,ARIMA


DATA_PATH = './dataset/'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
MAX_MAXWIND_SEQ_LEN = -1
MIN_LEN = 8
STATIONARIZATION='diff3'  # equal,diff1,diff2,diff3,diff4,log,sqrt,logdiff
p = 0
q = 0


# read dataset to dataframe
print('Loading data...')
# train_data_df = pd.read_csv(DATA_PATH + TRAIN_FILE, index_col=0)
# print(train_data_df.head())
test_data_df = pd.read_csv(DATA_PATH + TEST_FILE, index_col=0)
print(test_data_df.head())
print('Data load finish.')
print('----------------------------------------------------------------------------------------------------')

# convert dataframe from string to list
train_data = []
# for index, item in train_data_df.iterrows():
#     row = [item['ID']]
#     row.append(list(map(int, item['Date'][1:-1].split(','))))
#     row.append(list(map(int, item['Time'][1:-1].split(','))))
#     row.append(list(item['Event'][1:-1].split(',')))
#     row.append(list(item['Status'][1:-1].split(',')))
#     row.append(list(map(float, item['Latitude'][1:-1].split(','))))
#     row.append(list(map(float, item['Longitude'][1:-1].split(','))))
#     row.append(list(map(float, item['MaxWind'][1:-1].split(','))))
#     train_data.append(row)

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

        for i in range(MIN_LEN+1, item_seq_len):
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


# x_train, x_aux_month_train, x_aux_time_train, x_aux_lalo_train, x_aux_len_train, x_aux_stat_train, y_train = data2tensor(train_data, MAX_MAXWIND_SEQ_LEN)
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

# print('The train samples are' + str(len(y_train)) + '.')
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

# normalization
# x_train = x_train/max_maxWind
# x_train_rev = x_train_rev/max_maxWind
# x_train_lstm = x_train_lstm/max_maxWind
# x_aux_lalo_train = x_aux_lalo_train/90
# x_aux_stat_train = x_aux_stat_train/max_maxWind
# y_train = y_train/max_maxWind

# x_test = x_test/max_maxWind
# x_test_rev = x_test_rev/max_maxWind
# x_test_lstm = x_test_lstm/max_maxWind
# x_aux_lalo_test = x_aux_lalo_test/90
# x_aux_stat_test = x_aux_stat_test/max_maxWind
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

## Prepare the data-------------------------------------------------------------------------------------------------

# stationarization and show the curves
x_test_stationary = []
x_test_origin = []
x_test_diff1 = []
x_test_diff2 = []
x_test_diff3 = []
y_test_stationary = []
for i in range(0,len(x_test)):
    # equal,diff1,diff2,diff3,diff4,log,sqrt,logdiff
    if STATIONARIZATION=='equal':
        row_stationary = pd.Series(x_test[i])
        row_stationary.plot()
        new_row = row_stationary.tolist()
        if new_row != []:
            x_test_origin.append(x_test[i])
            x_test_stationary.append(new_row)
            y_test_stationary.append(y_test[i])
    if STATIONARIZATION=='diff1':
        row_stationary = pd.Series(x_test[i]).diff(1)
        row_stationary.plot()
        new_row = row_stationary.tolist()[1:]
        if new_row != []:
            x_test_origin.append(x_test[i])
            x_test_stationary.append(new_row)
            y_test_stationary.append(y_test[i])
    if STATIONARIZATION=='diff2':
        row_stationary = pd.Series(x_test[i]).diff(1)
        row_stationary2 = pd.Series(row_stationary).diff(1)
        row_stationary2.plot()
        new_row = row_stationary2.tolist()[2:]
        if new_row != []:
            x_test_origin.append(x_test[i])
            x_test_stationary.append(new_row)
            y_test_stationary.append(y_test[i])
    if STATIONARIZATION=='diff3':
        row_stationary = pd.Series(x_test[i]).diff(1)
        row_stationary2 = row_stationary.diff(1)
        row_stationary3 = row_stationary2.diff(1)
        row_stationary3.plot()
        x_test_origin.append(x_test[i])
        x_test_diff1.append(row_stationary.tolist()[1:])
        x_test_diff2.append(row_stationary2.tolist()[2:])
        x_test_diff3.append(row_stationary3.tolist()[3:])
        x_test_stationary.append(row_stationary3.tolist()[3:])
        y_test_stationary.append(y_test[i])
    if STATIONARIZATION=='diff4':
        row_stationary = pd.Series(x_test[i]).diff(1)
        row_stationary2 = pd.Series(row_stationary).diff(1)
        row_stationary3 = pd.Series(row_stationary2).diff(1)
        row_stationary4 = pd.Series(row_stationary3).diff(1)
        row_stationary4.plot()
        new_row = row_stationary4.tolist()[4:]
        if new_row != []:
            x_test_origin.append(x_test[i])
            x_test_stationary.append(new_row)
            y_test_stationary.append(y_test[i])
    if STATIONARIZATION=='sqrt':
        row_stationary = []
        for j in x_test[i]:
            row_stationary.append(np.sqrt(j))
        if row_stationary != []:
            x_test_origin.append(x_test[i])
            x_test_stationary.append(row_stationary)
            pd.Series(row_stationary).plot()
            y_test_stationary.append(y_test[i])
    if STATIONARIZATION=='log':
        row_stationary = []
        for j in x_test[i]:
            row_stationary.append(np.log(j))
        if row_stationary != []:
            x_test_origin.append(x_test[i])
            x_test_stationary.append(row_stationary)
            pd.Series(row_stationary).plot()
            y_test_stationary.append(y_test[i])
    if STATIONARIZATION=='logdiff':
        row_stationary = []
        for j in x_test[i]:
            row_stationary.append(np.log(j))
        row_stationary = pd.Series(row_stationary).diff(1)
        row_stationary.plot()
        new_row = row_stationary.tolist()[1:]
        if new_row != []:
            x_test_origin.append(x_test[i])
            x_test_stationary.append(new_row)
            y_test_stationary.append(y_test[i])
# plt.show()

# ADF
# in this part, ADF filter x_test_stationary to x_test_adf, here use x_test_stationary_idx to remember the index of the sequence pass ADF test in x_test_statioary
print('ADF filter:')
x_test_stationary_idx = []
x_test_adf = []
y_test_adf = []
for i in range(0,len(x_test_stationary)):
    t=sm.tsa.stattools.adfuller(x_test_stationary[i], maxlag=1)
    output=pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
    output['value']['Test Statistic Value'] = t[0]
    output['value']['p-value'] = t[1]
    output['value']['Lags Used'] = t[2]
    output['value']['Number of Observations Used'] = t[3]
    output['value']['Critical Value(1%)'] = t[4]['1%']
    output['value']['Critical Value(5%)'] = t[4]['5%']
    output['value']['Critical Value(10%)'] = t[4]['10%']
    # print(output)
    if t[0] <= t[4]['5%']:
        x_test_stationary_idx.append(i)
        x_test_adf.append(x_test_stationary[i])
        y_test_adf.append(y_test_stationary[i])
assert len(x_test_adf)==len(y_test_adf)
print('stationary sequence counts = ',len(x_test_adf))


prediction = []
for i in range(0,len(x_test_adf)):
    arma_model = ARMA(x_test_adf[i],(p,q)).fit(disp=-1,maxiter=100)
    predict_data = arma_model.predict(start=0, end=len(x_test_adf[i]), dynamic=False)
    idx = x_test_stationary_idx[i]  # the index of this sequence in x_test_stationary
    assert x_test_adf[i]==x_test_stationary[idx]
    if STATIONARIZATION=='diff3':
        prediction.append(x_test_origin[idx][-1]+x_test_diff1[idx][-1]+x_test_diff2[idx][-1]+predict_data[-1])
print('Ground Truth:',y_test_adf)
print('Prediction:',[round(i,1) for i in prediction])

# calculate RMSE
score = 0
n = len(y_test_adf)
for i in range(0,n):
    score = score + np.square(prediction[i]-y_test_adf[i])
score = np.sqrt(score/n)
print('RMSE=',score)

exit()
