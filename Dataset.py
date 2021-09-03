import tensorflow as tf
import pandas as pd
import glob
from Config import *
from collections import deque
import numpy as np
from sklearn.preprocessing import scale, MinMaxScaler
import pandas_ta as pta
from AutoTrader import *
from keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras import Sequential

"""
    Binance Dataset Format:
    ["Open time", "Open", "High", "Low", "Close", "Volume", "Close time", "Quote asset volume", "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]
    [    0          1       2       3       4       5           6               7                   8                   9                               10                             11]
"""

defaultNames = ["time", "Open", "High", "Low", "Close", "Volume", "CloseTime",
                "QuoteAssetVolume", "NumberOfTrades", "TBaseAssetVolume", "TQuoteAssetVolume", "Ignore"]


class Dataset:
    def __init__(self):
        print("> Dataset Class Initialized")
        self.path = f"datasets/{COIN_PEAR}/*.csv"
        all_files = sorted(glob.glob(self.path))

        print(f">> loading Data: {COIN_PEAR}")
        # Loading All CSV
        self.df = pd.concat(
            (pd.read_csv(f, header=None, names=defaultNames) for f in all_files))

        # Removing Unwanted Colums
        self.df.set_index('time', inplace=True)
        self.df = self.df[['Low', 'High', "Open", "Close", "Volume"]]
        # self.df.drop('CloseTime', axis=1, inplace=True)
        # self.df.drop('QuoteAssetVolume', axis=1, inplace=True)
        # self.df.drop('TBaseAssetVolume', axis=1, inplace=True)
        # self.df.drop('TQuoteAssetVolume', axis=1, inplace=True)
        # self.df.drop('Ignore', axis=1, inplace=True)
        print(f">> Data Loaded")
        self.scaler = MinMaxScaler()
        # print(self.df.head(10))
        # print(self.df[:20].values)
        #     li.append(df)
    # def loadDataset(self):

    def buy(self, current, future):
        if not self.doAction(current, future):
            return 0
        if(float(future) > float(current)):
            return 1
        else:
            return 0

    def sell(self, current, future):
        if not self.doAction(current, future):
            return 0
        if(float(future) < float(current)):
            return 1
        else:
            return 0

    def Action(self, current, future):
        if not self.doAction(current, future):
            return 0

        if(float(future) > float(current)):
            return 1
        else:
            return 0

    def doAction(self, current, future):
        if(float(future) > (float(current) + float(current) * DO_ACTION_MIN_CHANCE)):
            return 1
        elif (float(future) < (float(current) - float(current) * DO_ACTION_MIN_CHANCE)):
            return 1
        else:
            return 0

    def transformSquenceData(self, data):
        last = data[-1]  # Get last element
        x = []
        # Latest Values
        latest_low_price = last[0]
        latest_high_price = last[1]
        latest_open_price = last[2]
        latest_close_price = last[3]
        latest_volume = last[4]

        # Accelerations and Differences
        acc_index = 0
        total_window_volume = 0
        prev_window_volume_acceleration = 0
        window_volume_acceleration = 0
        # The average volume acceleration based on the feature window (local)
        average_volume_acceleration = 0
        # The average close price acceleration based on the feature window (local)
        average_close_price_acceleration = 0
        # The average close price difference based on the feature window (local)
        average_close_price_diff = 0
        # Averages
        average_close_price = 0
        average_volume = 0

    def add_indicators(self, df):
        # Get RSI
        df['RSI'] = pta.rsi(df['Close'], length=14)

        # GET EMA
        df['EMA'] = pta.ema(pta.ohlc4(df["Open"], df["High"],
                            df["Low"], df["Close"]), length=14)

        # Calculate SMA
        df['SMA'] = pta.sma(df["Close"], length=14)

        donchiandf = pta.donchian(
            self.df["High"], self.df["Low"], lower_length=10, upper_length=15)
        df = pd.merge(df, donchiandf, on='time')
        return df

    def add_predictions(self, df):
        df['future'] = df['Close'].shift(-FUTURE_PREDICT)
        df['buy'] = list(map(self.buy, self.df['Close'], df['future']))
        df['sell'] = list(map(self.sell, self.df['Close'], df['future']))
        del df['future']
        return df

    def split_data(self, df):
        df_train = self.df.loc[(df.index > TRAINING_START_TS) & (
            df.index < TRAINING_END_TS)]
        df_test = self.df.loc[(df.index > START_TS) & (df.index < END_TS)]
        return df_train, df_test

    def pct_change(self, df):

        for col in df.columns:
            if not ((col == "buy") or (col == "sell") or (col == "DCU_10_15") or (col == "DCU_10_15") or (col == "RSI") or (col == "Volume")):
                df[f"{col}_pct_change"] = df[col].pct_change()

        # print(df.columns)
        return df  # Ensure buy/sell stays last

    def scale_data(self, df):
        # print(df.columns[:len(df.columns)-2])

        df[[*df.columns[:len(df.columns)-2]]] = self.scaler.fit_transform(df[[*df.columns[:len(df.columns)-2]]])
        # df = self.scaler.fit_transform(df[df.columns[:len(df.columns)-2]])
        return df

    def make_seq(self, df):
        seq_data = []
        deq = deque(maxlen=LOOK_BACK_LEN)
        for index, row in df.iterrows():
            deq.append(row[:len(df.columns)-2].values)
            if len(deq) == LOOK_BACK_LEN:
                seq_data.append([list(deq), row[:len(df.columns)-2:].values])
        return seq_data



    def ProcessData(self):
        self.df.dropna(inplace=True)
        # self.df = self.add_indicators(self.df)
        # self.df.dropna(inplace=True)
        # self.df = self.pct_change(self.df)
        # self.df.dropna(inplace=True)
        self.df = self.add_predictions(self.df)
        self.df = self.scale_data(self.df)
        # print(self.df.head(200))
        # print(df_test.head(200))
        # trader = AutoTrader()
        return self.split_data(self.df)

        # trader.runSimulation(df_test)
        # self.df.to_csv("out2.csv")
