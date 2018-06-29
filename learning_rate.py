import sys
sys.path.insert(0,'/home/augusta/dev/autonomio_dev/autonomio')

from autonomio import wrangler, data

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l1_l2
from keras.optimizers import *

import numpy as np
from inspect import getargspec

df = data('../../datasets/mushrooms.csv', 'file')
mush = wrangler(df)
mush = np.array(mush)
mush = mush.astype(int)

def save_file(name, acc):
    
    f = open(name, 'w')
    
    for a in acc[:-1]:
        f.write(str(a) + ', ')
    f.write(str(acc[-1]))
    
    f.close

optimizers = [SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam]
names = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

lr_default = getargspec(optimizers[n].__init__).defaults[0]
lr_min = 0.001 * lr_default
lr_max = 10 * lr_default
r = np.linspace(lr_min, lr_max, 100).tolist()

if lr_default not in r:
    r.append(lr_default)
    r.sort()

acc = []
val_acc = []
drop = 0.5
act = 'linear'

num = len(r)
x = 0

for n in range(len(names))
    for lr in r:

        temp_acc = []
        temp_vacc = []

        model = Sequential()
        model.add(Dense(20, input_dim = 22, activation=act))
        model.add(LeakyReLU())
        model.add(Dropout(drop))
        model.add(Dense(20, activation=act))
        model.add(LeakyReLU())
        model.add(Dropout(drop))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='logcosh', optimizer=optimizers[n](lr=lr), metrics=['accuracy'])

        for j in range(3):
            h = model.fit(mush[:,1:], mush[:,0], validation_split=.33, epochs=50, batch_size=32, verbose=0)
            a = sum(h.history['acc'][-5:])/ 5
            b = sum(h.history['val_acc'][-5:])/ 5
            temp_acc.append(a)
            temp_vacc.append(b)

        a2 = sum(temp_acc)/3
        b2 = sum(temp_vacc)/3

        acc.append(a2)
        val_acc.append(b2)

        x += 1
        print('done: ' + str(x) + ' out of ' + str(num))
    

    name = names[n] + '.txt'
    save_file(name, acc)

    val_name = 'val_' + names[n] + '.txt'
    save_file(val_name, val_acc)
