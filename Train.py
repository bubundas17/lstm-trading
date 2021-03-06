from  Dataset import *

dataset = Dataset()

train, test = dataset.ProcessData()

# dataset.save_scaling()

from keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras import Sequential
import tensorflow as tf

ts_generator = TimeseriesGenerator(train[[*train.columns[:len(train.columns)-2]]].values, train[[*train.columns[-1:]]].values, length=LOOK_BACK_LEN, sampling_rate=1, batch_size=500)
generator_test = TimeseriesGenerator(test[[*test.columns[:len(test.columns)-2]]].values, test[[*test.columns[-1:]]].values, length=LOOK_BACK_LEN, sampling_rate=1, batch_size=20)

# print(pd.array(ts_generator[0][0][0]).shape)

model = Sequential()
model.add(LSTM(128, input_shape=(360,len(ts_generator[0][0][0][0])), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='mean_squared_error',
    optimizer=opt,
    metrics=['accuracy']
)

model.fit(
    ts_generator,
    epochs=5,
    # validation_data = generator_test,
    shuffle=True
)

model.evaluate(generator_test)
model.save("model")
# model.predict(generator_test)

# ts_generator = TimeseriesGenerator(df_train[[*df_train.columns[:len(df_train.columns)-2]]].values, df_train[[*df_train.columns[-2:]]].values, length=LOOK_BACK_LEN, sampling_rate=1, batch_size=1)
# print(df_train.head())
# print(ts_generator[0])


# model = Sequential()

# model.add(LSTM(512, activation = "relu", return_sequences=True, input_shape = (LOOK_BACK_LEN, 19)))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())

# model.add(LSTM(256,  activation="relu"))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())

# model.add(LSTM(128, activation="relu"))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())

# model.add(LSTM(128, activation="relu"))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())

# model.add(Dense(128, activation="relu"))
# model.add(Dropout(0.2))

# model.add(Dense(64, activation="relu"))
# model.add(Dropout(0.2))

# model.add(Dense(2, activation="softmax"))
# optimizer = tf.keras.optimizers.Adam(lr = 0.001, decay = 1e-6)
# model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizer)

# model.fit(ts_generator, epochs=500, verbose=0)