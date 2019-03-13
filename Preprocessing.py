import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = './dataset/'
DATASET_FILE = 'pacific.csv'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
DRAW_COUNTS = False
DRAW_MAXWINDS_1 = False
DRAW_MAXWINDS_ALL = True
FIX_LEN = 19


def draw_hist(data_dict, bins, title, xlabel, ylabel, xmin, xmax, ymin, ymax):
    hist_list = []
    for _k in data_dict.keys():
        hist_list.append(len(data_dict[_k][1]))  # v[1] is Date

    plt.hist(hist_list, bins)
    plt.xlabel(xlabel)
    plt.xlim(xmin, xmax)
    plt.ylabel(ylabel)
    plt.ylim(ymin, ymax)
    plt.title(title)
    plt.show()


# def draw_curve(x_data, y_data, title, xlabel, ylabel, xmin, xmax, ymin, ymax):
#     plt.plot(x_data, y_data, mec='r', mfc='w', label='curve')
#     plt.legend()
#     plt.xlabel(xlabel)
#     plt.xlim(xmin, xmax)
#     plt.ylabel(ylabel)
#     plt.ylim(ymin, ymax)
#     plt.title(title)
#     plt.show()
#
#
# def process_len(data_dict, file_name):
#     data_list = []
#     for dict_key in data_dict:
#         if len(data_dict[dict_key]) == FIX_LEN:
#             data_list.append(data_dict[dict_key])
#         if len(data_dict[dict_key]) < FIX_LEN:
#             temp_list = []
#             j = 0
#             for i in range(FIX_LEN):
#                 if j >= (len(data_dict[dict_key])-1):
#                     break
#                 while round((j+1)/(len(data_dict[dict_key])-1), 2) < round(i/(FIX_LEN-1), 2):
#                     j += 1
#                 left = round(i / (FIX_LEN - 1), 2) - round(j / (len(data_dict[dict_key]) - 1), 2)
#                 right = round((j + 1) / (len(data_dict[dict_key]) - 1), 2) - round(i / (FIX_LEN - 1), 2)
#                 num = data_dict[dict_key][j] * round(right / (left + right), 2) + data_dict[dict_key][j + 1] * round(
#                     left / (left + right), 2)
#                 temp_list.append(round(num, 2))
#             data_list.append(temp_list)
#         if len(data_dict[dict_key]) > FIX_LEN:
#             temp_list = []
#             j = 0
#             for i in range(FIX_LEN):
#                 if j >= (len(data_dict[dict_key]) - 1):
#                     break
#                 while round((j + 1) / (len(data_dict[dict_key]) - 1), 2) < round(i / (FIX_LEN - 1), 2):
#                     j += 1
#                 left = round(i / (FIX_LEN - 1), 2) - round(j / (len(data_dict[dict_key]) - 1), 2)
#                 right = round((j + 1) / (len(data_dict[dict_key]) - 1), 2) - round(i / (FIX_LEN - 1), 2)
#                 num = data_dict[dict_key][j] * round(right / (left + right), 2) + data_dict[dict_key][j + 1] * round(
#                     left / (left + right), 2)
#                 temp_list.append(round(num, 2))
#             data_list.append(temp_list)
#     name = range(FIX_LEN)
#     csv_train = pd.DataFrame(columns=name, data=data_list)
#     csv_train.to_csv(DATA_PATH+file_name)


# read dataset csv
print('Loading data...')
csv_data = pd.read_csv(DATA_PATH + DATASET_FILE)
print(csv_data.head())
print('Data load finish.')
print('----------------------------------------------------------------------------------------------------')

# process dataset from dataframe to dictionary {ID:list of lists}
dataset_dict = {}  # format:ID:[date,time,event,status,latitude,longitude,maxWind]
date_list = []
time_list = []
event_list = []
status_list = []
latitude_list = []
longitude_list = []
maxWind_list = []
for index, row in csv_data.iterrows():
    if row['ID'] not in dataset_dict.keys():
        dataset_dict[row['ID']] = [[], [], [], [], [], [], []]
    dataset_dict[row['ID']][0].append(row['Date'])
    dataset_dict[row['ID']][1].append(row['Time'])
    dataset_dict[row['ID']][2].append(row['Event'])
    dataset_dict[row['ID']][3].append(row['Status'])
    dataset_dict[row['ID']][4].append(float(row['Latitude'][:-1]))  # get rid of the 'N' in latitude
    dataset_dict[row['ID']][5].append(float(row['Longitude'][:-1]))  # get rid of the 'W' in longitude
    dataset_dict[row['ID']][6].append(float(row['Maximum Wind']))
print('Total '+str(len(dataset_dict))+' items.')

# draw histogram
if DRAW_COUNTS:
    draw_hist(dataset_dict, 121, 'HU and TY data point counts hist', 'counts each HU/TY', 'counts', 1, 121, 0, 55)
# IDs and max wind sequences
if DRAW_MAXWINDS_ALL:
    # draw all curves
    for key in dataset_dict.keys():
        plt.plot(range(len(dataset_dict[key])), dataset_dict[key])
    plt.xlabel('time')
    plt.xlim(0, 120)
    plt.ylabel('max wind')
    plt.ylim(0, 200)
    plt.title('All curves')
    plt.show()

# split train and test
train_dict = {}
test_dict = {}
for key in dataset_dict:
    if key[4:] == '2015' or key[4:] == '2014':
        test_dict[key] = dataset_dict[key]
    else:
        train_dict[key] = dataset_dict[key]
print('Train ' + str(len(train_dict)) + ' items.')
print('Test ' + str(len(test_dict)) + ' items.')
print('----------------------------------------------------------------------------------------------------')

# save the tran/test dataset into csv file
cols = ['ID', 'Date', 'Time', 'Event', 'Status', 'Latitude', 'Longitude', 'MaxWind']
# train.csv
train_list = []
for k, v in train_dict.items():
    row = [k, v[0], v[1], v[2], v[3], v[4], v[5], v[6]]
    train_list.append(row)
train_df = pd.DataFrame(data=train_list, index=None, columns=cols)
train_df.to_csv(DATA_PATH + TRAIN_FILE)
print('Train dataset has been saved in ' + DATA_PATH + TRAIN_FILE + '.')
# test.csv
test_list = []
for k, v in test_dict.items():
    row = [k, v[0], v[1], v[2], v[3], v[4], v[5], v[6]]
    test_list.append(row)
test_df = pd.DataFrame(data=test_list, index=None, columns=cols)
test_df.to_csv(DATA_PATH + TEST_FILE)
print('Test dataset has been saved in ' + DATA_PATH + TEST_FILE + '.')
print('----------------------------------------------------------------------------------------------------')


#
# # one ID and its curve
# if DRAW_MAXWINDS_1:
#     tmp_list = []
#     for index, row in csv_data.iterrows():
#         if row['ID'] == 'EP101994':
#             temp =
#             [row['ID'], row['Name'], row['Date'], row['Time'], row['Event'], row['Status'], row['Maximum Wind']]
#             tmp_list.append(temp)
#     for item in tmp_list:
#         print(item)
#
#     list_curve = []
#     for item in tmp_list:
#         list_curve.append(item[6])
#     print(list_curve)
#     x = range(len(list_curve))
#     draw_curve(x, list_curve, 'EP101994', 'time', 'max wind', 0, 125, 0, 200)

# process data to the same FIX_LEN
# process_len(train_dict, 'train19.csv')
# print('The training data has been saved.')
# process_len(test_dict, 'test19.csv')
# print('The testing data has been saved.')
