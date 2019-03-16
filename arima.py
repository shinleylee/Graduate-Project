import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

DATA_PATH = './dataset/'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
FIX_LEN = 19
DRAW_CURVES = True
TRAIN_LEN = 4
EMBEDDING_DIM = 32

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


# process the data into tensors
def data2tensor(dataset):
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
            # maxWind
            x.append(maxWind_seq[:i])
            y.append(maxWind_seq[i])

    return x, y


x_test, y_test = data2tensor(test_data)
assert len(x_test)==len(y_test)

print('The test samples are ' + str(len(y_test)) + '.')
print('----------------------------------------------------------------------------------------------------')

## Prepare the data-------------------------------------------------------------------------------------------------

# 移动平均图
def draw_trend(timeseries, size):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeseries.rolling(window=size).mean()
    # 对size个数据移动平均的方差
    rol_std = timeseries.rolling(window=size).std()
    timeseries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_std.plot(color='black', label='Rolling standard deviation')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

def draw_ts(timeseries):
    f = plt.figure(facecolor='white')
    timeseries.plot(color='blue')
    plt.show()

# Dickey-Fuller test:
def teststationarity(ts):
    dftest = sm.tsa.stattools.adfuller(ts)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    return dfoutput

def decompose(timeseries):
    # 返回包含三个部分 trend（趋势部分） ， seasonal（季节性部分） 和residual (残留部分)
    decomposition = seasonal_decompose(timeseries, freq=4)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    plt.subplot(411)
    plt.plot(x_log, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    return trend, seasonal, residual

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def draw_acf_pacf(ts, lags):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts, ax=ax1, lags=lags)
    ax2 = f.add_subplot(212)
    plot_pacf(ts, ax=ax2, lags=lags)
    plt.subplots_adjust(hspace=0.5)
    plt.show()






for x in x_test:
    x_log = np.log(pd.Series(x))
    if x_log.shape[0] >= 12:
    #     draw_trend(x_log, 4)
    #     draw_ts(x_log)
    #     teststationarity(x_log)
    #
        x_log.index = pd.to_datetime(x_log.index)
    #     print(x_log)
    #     trend, seasonal, residual = decompose(x_log)
    #     residual.dropna(inplace=True)
    #     draw_trend(residual, 12)
    #     teststationarity(residual)

        rol_mean = x_log.rolling(window=4).mean()
        rol_mean.dropna(inplace=True)
        x_diff_1 = rol_mean.diff(1)
        x_diff_1.dropna(inplace=True)
        # teststationarity(x_diff_1)

        x_diff_2 = x_diff_1.diff(1)
        x_diff_2.dropna(inplace=True)
        # teststationarity(ts_diff_2)

        # get the coef by observing the graph
        # draw_acf_pacf(x_diff_2, 4)

        # make prediction
        from statsmodels.tsa.arima_model import ARIMA
        model = ARIMA(x_log, order=(1, 1, 1))
        result_arima = model.fit(disp=-1, method='css')

        # recover the data
        predict_ts = result_arima.predict()
        # 一阶差分还原
        diff_shift_ts = x_diff_1.shift(1)
        diff_recover_1 = predict_ts.add(diff_shift_ts)
        # 再次一阶差分还原
        rol_shift_ts = rol_mean.shift(1)
        diff_recover = diff_recover_1.add(rol_shift_ts)
        # 移动平均还原
        rol_sum = x_log.rolling(window=11).sum()
        rol_recover = diff_recover * 12 - rol_sum.shift(1)
        # 对数还原
        log_recover = np.exp(rol_recover)
        log_recover.dropna(inplace=True)

        # mse
        x = x_log[log_recover.index]
        # 过滤没有预测的记录
        plt.figure(facecolor='white')
        log_recover.plot(color='blue', label='Predict')
        x.plot(color='red', label='Original')
        plt.legend(loc='best')
        plt.title('RMSE: %.4f'% np.sqrt(sum((log_recover-x)**2)/x.size))
        plt.show()

print('end')
