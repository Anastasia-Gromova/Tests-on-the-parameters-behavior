import sys
sys.path.insert(0,'/home/augusta/dev/autonomio_dev/autonomio')

from autonomio import wrangler, data

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l1_l2

import numpy as np

df = data('../../datasets/train.csv', 'file')

# cleaning the data

mush = wrangler(df)
mush = np.array(mush)
mush = mush.astype(int)

# function to save the file

def save_file(name, acc):
    
    f = open(name, 'w')
    
    for a in acc[:-1]:
        f.write(str(a) + ', ')
    f.write(str(acc[-1]))
    
    f.close

act = 'linear'
drop = 0.5
nodes = range(5, 50, 1)
layers = [1, 2, 3, 5, 10, 20, 30]
k = 0
acc = []

n = len(nodes) * len(layers)
x = 0

for la in layers:
    for node in nodes:

        l = []
        
        # training the data
        
        model = Sequential()
        model.add(Dense(node, input_dim = 22, activation=act))
        model.add(LeakyReLU())
        model.add(Dropout(drop))

        for i in range(la):
            model.add(Dense(node / 2, activation=act))
            model.add(LeakyReLU())
            model.add(Dropout(drop))

        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='logcosh', optimizer='RMSProp', metrics=['accuracy'])

        for j in range(3):
            h = model.fit(mush[:,1:], mush[:,0], validation_split=.33, epochs=50, batch_size=32, verbose=0)
            a = sum(h.history['val_acc'][-10:])/10
            l.append(a)

        b = sum(l) / 3
        acc.append(b)

        x += 1
        print('done: ' + str(x) + ' out of ' + str(n))
    

    name = str(la) + '_layers.txt'
    save_file(name, acc)
    
