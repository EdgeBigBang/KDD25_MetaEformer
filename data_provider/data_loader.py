import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

class Dataset_ECL(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ECL_57.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 48
            self.label_len = 12
            self.pred_len = 24
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_raw = df_raw[['date'] + cols + [self.target]]
        df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Traffic(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                  data_path='Traffic_852.csv',target='OT',
                 scale=True, timeenc=0, freq='h'):

        if size == None:
            self.seq_len = 48
            self.label_len = 12
            self.pred_len = 24
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        num_train = int(len(df_raw) * 0.8)
        num_test = int(len(df_raw) * 0.1)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_raw = df_raw[['date'] + cols + [self.target]]
        df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Cloud(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='ECW.npy', target=0, scale=True,
                 timeenc=0, freq='d', step=12):

        if size == None:
            self.seq_len = 48
            self.label_len = 12
            self.pred_len = 24
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.scaler = 0

        self.root_path = root_path
        self.data_path = data_path
        self.step = step
        self.__read_data__()

    def __read_data__(self):

        X_all = np.load(open(os.path.join(self.root_path, self.data_path), 'rb'), allow_pickle=True)
        y = X_all[:, :, 0]
        num_ts, num_periods, num_features = X_all.shape

        num_train = int(num_periods * 0.8)
        num_test = int(num_periods * 0.2)

        border1s = [0, 0, num_periods - num_test]
        border2s = [num_train, 0, num_periods]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        X = X_all[:, border1:border2, :]
        y = y[:, border1:border2]
        if self.set_type == 2:
            Xtr = X_all[:, 0:num_train, :]
            print('ceshi')

        num_ts, num_periods, num_features = X.shape
        xscaler = MinMaxScaler()
        yscaler = MinMaxScaler()

        yscaler.fit(y.reshape(-1, 1))
        self.scaler = yscaler

        if self.set_type == 2:
            print('ceshi_1')
            Xtr = xscaler.fit_transform(Xtr.reshape(-1, num_features)).reshape(num_ts, -1, num_features)
            X = xscaler.transform(X.reshape(-1, num_features)).reshape(num_ts, -1, num_features)
        else:
            X = xscaler.fit_transform(X.reshape(-1, num_features)).reshape(num_ts, -1, num_features)

        X_train_all = []
        Y_train_all = []
        X_mark_all = []
        Y_mark_all = []


        for i in range(num_ts):
            for j in range(self.seq_len, num_periods - self.pred_len, self.step):
                X_train_all.append(X[i, j - self.seq_len:j, 0])
                Y_train_all.append(X[i, j - self.label_len:j + self.pred_len, 0])
                X_mark_all.append(X[i, j - self.seq_len:j, 1:])
                Y_mark_all.append(X[i, j - self.label_len:j + self.pred_len, 1:4])

        self.X = np.asarray(X_train_all).reshape(-1, self.seq_len, 1)
        self.Y = np.asarray(Y_train_all).reshape(-1, self.label_len + self.pred_len, 1)
        self.X_mark = np.asarray(X_mark_all).reshape(-1, self.seq_len, 15)
        self.Y_mark = np.asarray(Y_mark_all).reshape(-1, self.label_len + self.pred_len, 3)

    def __len__(self):
        return self.X.shape[0]

    def get_scalse(self):
        return self.scaler

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.X_mark[index], self.Y_mark[index]

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 48
            self.label_len = 12
            self.pred_len = 24
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
        else:
            cols = list(df_raw.columns)
            self.cols = cols.copy()
            cols.remove('date')
        cols.remove(self.target)
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)


        df_raw = df_raw[['date'] + cols + [self.target]]
        df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        self.future_dates = list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ECW():
    def __init__(self, X, enc_len=48, label_len=12, pred_len=24, step=12):
        num_ts, num_periods, num_features = X.shape
        X_train_all = []
        Y_train_all = []
        X_mark_all = []
        Y_mark_all = []

        # 滑动窗口
        for i in range(num_ts):
            for j in range(enc_len, num_periods - pred_len, step):
                X_train_all.append(X[i, j - enc_len:j, 0])
                Y_train_all.append(X[i, j - label_len:j + pred_len, 0])
                X_mark_all.append(X[i, j - enc_len:j, 1:])  # 携带静态特征
                Y_mark_all.append(X[i, j - label_len:j + pred_len, 1:4])

        self.X = np.asarray(X_train_all).reshape(-1, enc_len, 1)
        self.Y = np.asarray(Y_train_all).reshape(-1, label_len + pred_len, 1)
        self.X_mark = np.asarray(X_mark_all).reshape(-1, enc_len, 15)
        self.Y_mark = np.asarray(Y_mark_all).reshape(-1, label_len + pred_len, 3)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.X_mark[index], self.Y_mark[index]

class Cloud_Dataset_zeroshot(Dataset):
    def __init__(self, X, enc_len=48, label_len=12, pred_len=24, step=12):
        num_ts, num_periods, num_features = X.shape
        X_train_all = []
        Y_train_all = []
        X_mark_all = []
        Y_mark_all = []

        for i in range(num_ts):
            for j in range(enc_len, num_periods - pred_len, step):
                X_train_all.append(X[i, j - enc_len:j, 0])
                Y_train_all.append(X[i, j - label_len:j + pred_len, 0])
                X_mark_all.append(X[i, j - enc_len:j, 1:])  # 携带静态特征
                Y_mark_all.append(X[i, j - label_len:j + pred_len, 1:4])

        self.X = np.asarray(X_train_all).reshape(-1, enc_len, 1)
        self.Y = np.asarray(Y_train_all).reshape(-1, label_len + pred_len, 1)
        self.X_mark = np.asarray(X_mark_all).reshape(-1, enc_len, 15)
        self.Y_mark = np.asarray(Y_mark_all).reshape(-1, label_len + pred_len, 3)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.X_mark[index], self.Y_mark[index]
