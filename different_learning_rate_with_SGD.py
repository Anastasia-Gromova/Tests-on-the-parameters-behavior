import sys
sys.path.insert(0,'/home/augusta/dev/autonomio_dev/autonomio')

from inspect import getargspec
import numpy as np
import pandas as pd
import random
import datetime as dt

from autonomio import wrangler, train
from keras.optimizers import *
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense

titanic = pd.read_csv('~/dev/autonomio_dev/dev/datasets/train.csv')

l=[]

age = int(titanic.Age.mean())
std = int(titanic.Age.std())
lo = int(age - std)
hi = int(age + std)

c = len(titanic.Age)

for i in range(c):
    
    if np.isnan(titanic.Age[i]) == True:
        
        l.append(random.randint(lo,hi))
    
    else:
        l.append(titanic.Age[i])
        
titanic.Age = l

df = wrangler(titanic,
              y='Survived',
              fill_columns='Cabin',
              starts_with_col='Cabin',
              nan_treshold=.8)

data = np.array(df)
data = data.astype(int)

optimizers = [SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam]
names = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

def train(data, batch):
    
    a = data.shape[1]-1
    dataY = data[:,1]
    dataX = data[:,2:a]
    
    model = Sequential()
    model.add(Dense(20, input_dim = dataX.shape[1], activation='elu'))
    model.add(Dense(20, activation='elu'))
    model.add(Dense(20, activation='elu'))
    model.add(Dense(20, activation='elu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='logcosh', optimizer='adam', metrics=['accuracy'])
    result = model.fit(dataX, dataY, validation_split=.33, epochs=150, batch_size=batch, verbose=0)
    
    return result

j = 0
epoch_time = 0

def clear():
    os.system('clear')

lr_default = getargspec(optimizers[0].__init__).defaults[0] # !
lr_min = 0.001 * lr_default
lr_max = 10 * lr_default

r = np.linspace(lr_min, lr_default, 50).tolist()
for i in np.linspace(lr_default + 0.0001, lr_max, 50).tolist():
    r.append(i)
    
if lr_default not in r:
    r.append(lr_default)
    r.sort()

acc = []
loss = []

counter = len(r) * 5
tests = str(counter)

for lr in r:

    temp_acc = []
    temp_loss = []

    start_time = dt.datetime.now()

    for k in range(5):

        history = train(data, 10)

        temp_acc.append(history.history['val_acc'][-3].mean())
        temp_loss.append(history.history['val_loss'][-3].mean())
        
        print (temp_acc)
        print (history.history['acc'][-3].mean())

        j += 1

    train_time = dt.datetime.now()
    epoch_time += (long(train_time.strftime('%s')) - long(start_time.strftime('%s'))) * counter
    finish_time = dt.datetime.fromtimestamp(epoch_time / j + long(start_time.strftime('%s')))
    finish_time = finish_time.strftime('%H:%M %d %B')
    
    clear()
    
    print('SGD 7/7')
    print('Estimated finish: ' + finish_time)
    counter -= 5
    print (str(j) + ' out of ' + tests + ' tests done')

    acc.append(sum(temp_acc) / 5)
    loss.append(sum(temp_loss) / 5)

f1 = open('lr_range_' + names[0] + '_.txt', 'w')   # !
for a in r:
    f1.write(str(a) + ', ')
f1.close()

f2 = open('lr_acc_' + names[0] + '_.txt', 'w')   # !
for b in acc:
    f2.write(str(b) + ', ')
f2.close()

f3 = open('lr_loss_' + names[0] + '_.txt', 'w')   # !
for c in loss:
    f3.write(str(c) + ', ')
f3.close()
