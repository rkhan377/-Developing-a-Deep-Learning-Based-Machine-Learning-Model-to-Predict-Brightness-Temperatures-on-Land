#by Rida Khan, Intern, Cooperative Institute for Satellite Earth System Studies
#This study was supported by NOAA grant NA19NES4320002 (Cooperative Institute for Satellite Earth System Studies -CISESS) at the University of Maryland/ESSIC.


import os

import time
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import LeakyReLU
from sklearn.preprocessing import StandardScaler

import math
import mpmath

def outlierAdd(ind, list):
    if ind not in list:
        list.append(ind)

LEARNING_RATE_DECAY = 0.99
OUTPUT_NODE = 22
RATIO_DATA=0.1
REGULARIZATION_RATE=0.01

INPUT_NODE = 210
BATCH_SIZE = 1024
LEARNING_RATE_BASE = 0.001
LAYERS_NODE = np.array([512, 384, 64])

tdir='D:\CISESS Data\original model relu\A'#file path that checkpoints, log and plot are saved to
wbfile="Amm.epoch99-acc0.9725230-loss0.0001724.hdf5" #load best model
npz=np.load("D:\CISESS Data\land17.npz") #input data file loc

#assigning input and label data
train_x=npz['datax']
train_y=npz['datay']

#longitue and latitude
lat = npz['lat']
lon = npz['lon']
#data preprocessing


#removing rows with 1 in first col/removing ocean data
list = [] #create list of rows so we can remove from x and y data
index = 0
for x in train_x:
        if x[0] == 17:
                list.append(index)
        index = index+1

print("remove ocean")
print(len(list))
train_x = np.delete(train_x, list, 0)
train_y = np.delete(train_y, list, 0)
lat = np.delete(lat, list, 0)
lon = np.delete(lon, list, 0)
print(train_x.shape, train_y.shape)


#trigonometry for sensor zenith angle, sensor scan angle, sensor azimuth angle

train_x=train_x[:,1:211] #capture only the cols 1-211

for row in train_x:
    row[1] = (mpmath.sec(math.radians(row[1])))
    row[2] = (mpmath.sec(math.radians(row[2])))
    row[3] = (mpmath.cos(math.radians(row[3])))



#removing nan
rowlist = []
rowi = 0
for row in train_x:
    if np.isnan(row).any(axis=0):
        rowlist.append(rowi)
    rowi = rowi+1


train_x = np.delete(train_x, rowlist, 0)
train_y = np.delete(train_y, rowlist, 0)
lat = np.delete(lat, rowlist, 0)
lon = np.delete(lon, rowlist, 0)
print("removed nans")
print(train_x.shape, train_y.shape)
#remove extreme y values
ylist = []
rowi = 0
for row in train_y:
    if np.any((row < 100) | (row > 350), axis=0):
        ylist.append(rowi)
    rowi = rowi+1


train_x = np.delete(train_x, ylist, 0)
train_y = np.delete(train_y, ylist, 0)
lat = np.delete(lat, ylist, 0)
lon = np.delete(lon, ylist, 0)
print("removed ys")
print(train_x.shape, train_y.shape)

print(train_x.shape, train_y.shape)

#normalize input
std=StandardScaler()
x_std=std.fit_transform(train_x)
train_x=x_std

# normalize output
y_std=std.fit_transform(train_y)
train_y=y_std

#shuffling
indices = tf.range(start=0, limit=tf.shape(train_x)[0], dtype=tf.int32)
idx = tf.random.shuffle(indices)
train_x = tf.gather(train_x, idx)
train_y = tf.gather(train_y, idx)
lat = tf.gather(lat, idx)
lon = tf.gather(lon, idx)

#save for graphs to retain shuffling
#np.save(tdir+"train_x shuffled",train_x)
#np.save(tdir+"train_y shuffled",train_y)
#np.save(tdir+"lat shuffled",lat)
#np.save(tdir+"lon shuffled",lon)

train_x = train_x[0:515332, :] #80%
train_y = train_y[0:515332, :]





ctime1=time.process_time()
rtime1=datetime.now()

# model architecture
model= Sequential()

ii=0
for nnodes in LAYERS_NODE:
    if ii==0:
        model.add(Dense(nnodes, input_dim=INPUT_NODE, kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=2.0/np.sqrt(nnodes))))  #Create input

        ii=ii+1
    else:
        model.add(Dense(nnodes, kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=2.0/np.sqrt(nnodes))))  #Create 3 hidden layer
    model.add(LeakyReLU(alpha=0.01))
    #model.add(Activation('relu'))
    #model.add(BatchNormalization())

model.add(Dense(OUTPUT_NODE, kernel_initializer='normal', kernel_regularizer=keras.regularizers.l2(REGULARIZATION_RATE), activation='linear')) #output layer

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    LEARNING_RATE_BASE,
    decay_steps=int(train_x.shape[0]) // BATCH_SIZE,
    decay_rate=LEARNING_RATE_DECAY,
    staircase=True)

opt=keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(loss='mse', optimizer=opt, metrics=['acc', 'mae'])  #1st dem mse, 2st dem mae, or any other function def

ctime2=time.process_time()
rtime2=datetime.now()

if os.path.isfile(wbfile):
    model.load_weights(wbfile)

filepath=tdir+'mm.epoch{epoch:02d}-acc{val_acc:.7f}-loss{val_loss:.7f}.hdf5'
checkpoint=ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callback_es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

csv_logger=CSVLogger(tdir+'log.csv', append=False)


ctime3=time.process_time()
rtime3=datetime.now()

model.fit(train_x, train_y,
          batch_size=BATCH_SIZE,
          shuffle=True,
          epochs=100,
          verbose=1,
          validation_split=RATIO_DATA,
          callbacks=[checkpoint, callback_es, csv_logger])

ctime4=time.process_time()
rtime4=datetime.now()

print("cpu time--", ctime2-ctime1, ctime3-ctime2, ctime4-ctime3)
print("running time--", rtime2-rtime1, rtime3-rtime2, rtime4-rtime3)