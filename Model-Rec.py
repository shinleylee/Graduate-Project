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
FIX_LEN = 19
DRAW_CURVES = True

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


# process the data into tensors
def data2tensor(dataset):
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
            maxWind_seq_5 = [(x-10)/5 for x in maxWind_seq]
            if i == 1:
                x.append([maxWind_seq_5[0]])
            elif i == 2:
                x.append([maxWind_seq_5[0]*100+maxWind_seq_5[1]])
            else:
                x.append([maxWind_seq_5[i-3]*10000+maxWind_seq_5[i-2]*100+maxWind_seq_5[i-1]])
            y.append([maxWind_seq[i]])

    for item in x:
        x_aux_len.append([1-math.log(len(item))])
        sum = 0
        for i in item:
            sum = sum+i
        x_aux_stat.append([sum/len(item)])

    return x, x_aux_month, x_aux_time, x_aux_lalo, x_aux_len, x_aux_stat, y


x_train, x_aux_month_train, x_aux_time_train, x_aux_lalo_train, x_aux_len_train, x_aux_stat_train, y_train = data2tensor(train_data)
x_test, x_aux_month_test, x_aux_time_test, x_aux_lalo_test, x_aux_len_test, x_aux_stat_test, y_test = data2tensor(test_data)
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

maxWind_li_input = [[x] for x in range(10,190,5)]

user_train_pos = x_train
item_train_pos = y_train
rate_train_pos = [1 for x in range(0,x_train.shape[0],1)]

user_train_neg = x_train.tolist()
user_train_neg_new = []
for item in user_train_neg:
    user_train_neg_new.append(item)
    user_train_neg_new.append(item)
    user_train_neg_new.append(item)
    user_train_neg_new.append(item)
user_train_neg = np.array(user_train_neg_new, dtype='int32')
item_train_neg = []
for num in item_train_pos:
    item_train_neg.append(num - 5)
    item_train_neg.append(num - 10)
    item_train_neg.append(num + 5)
    item_train_neg.append(num + 10)
item_train_neg = np.array(item_train_neg, dtype='int32')
rate_train_neg = [0 for x in range(0,x_train.shape[0]*4,1)]

user_train = np.concatenate((user_train_pos, user_train_neg),axis=0)
item_train = np.concatenate((item_train_pos, item_train_neg),axis=0)
rate_train = np.array(rate_train_pos + rate_train_neg)

user_test = x_test.tolist()
user_test_new = []
for item in user_test:
    for i in range(0,33,1):
        user_test_new.append(item)
user_test = np.array(user_test_new)
item_test = []
for i in range(0,x_test.shape[0],1):
    for item in maxWind_li_input:
        item_test.append(item)
item_test = np.array(item_test)

rate_test = np.array(y_test)

## Prepare the data-------------------------------------------------------------------------------------------------

def create_model():
    #输入数据的shape为(n_samples, timestamps, features)
    user_input = Input(shape=(1,), dtype='int32',name='user_input')
    user_em8 = Embedding(input_dim=370000, output_dim=8, input_length=1)(user_input)
    user_em8 = Flatten()(user_em8)
    user_em16 = Embedding(input_dim=370000, output_dim=16, input_length=1)(user_input)
    user_em16 = Flatten()(user_em16)
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    item_em8 = Embedding(input_dim=200, output_dim=8,  input_length=1)(item_input)
    item_em8 = Flatten()(item_em8)
    item_em16 = Embedding(input_dim=200, output_dim=16, input_length=1)(item_input)
    item_em16 = Flatten()(item_em16)

    # masking = Masking(mask_value=0)(main_input_lstm)
    # em_lstm = Embedding(input_dim=200, output_dim=8, input_length=MAX_MAXWIND_SEQ_LEN, mask_zero=True)(main_input)
    # lstm = LSTM(4)(em_lstm)

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

    gmf = Multiply()([user_em16, item_em16])
    mlp = Concatenate()([user_em8, item_em8])
    mlp = Dense(16, activation='relu')(mlp)

    ncf = Concatenate()([gmf, mlp])
    ncf = Dense(16, activation='relu')(ncf)
    main_output = Dense(1, activation='sigmoid', name='finalDense')(ncf)

    #下面还有个lstm，故return_sequences设置为True
    # model.add(Masking(mask_value=0, input_shape=(MAX_MAXWIND_SEQ_LEN, 1)))
    # model.add(LSTM(units=1, return_sequences=False, activation='linear'))
    # model.add(LSTM(units=1, activation='linear'))
    # model.add(Dense(units=1, activation='linear'))

    model = Model(inputs=[user_input, item_input], outputs=main_output)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def mse_weighted(y_true, y_pred):
    y_diff = y_true - y_pred
    weight = 100 * y_diff
    return Kbe.mean(weight*Kbe.square(y_pred - y_true), axis=-1)


model = create_model()
model.fit([user_train, item_train], rate_train,
          batch_size=128, epochs=2, validation_split=0.1, verbose=2)

# make predictions
testPredict = model.predict([user_test, item_test])
testPredict = np.reshape(testPredict,(testPredict.shape[0]))
print(testPredict)

y_pred = []

cur_user = -1
for i in range(0, user_test.shape[0]):
    maxSigmoid = 0
    maxPredictWind = 0
    for j in range(10,190,5):
        if maxSigmoid < testPredict[i*37+(j-10)/5]:
            maxSigmoid = testPredict[i*37+(j-10)/5]
            maxPredictWind = maxWind_li_input[math.ceil((j-10)/5)]
    y_pred.append(maxPredictWind)

print(rate_test)
print(y_pred)




exit()





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
