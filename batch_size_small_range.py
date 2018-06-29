import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense
import math
import datetime as dt

def train(data, batch):
    
    # Separating dataset on the input (dataX) and output (dataY)
    a = data.shape[1]-1
    dataY = data[:,1]
    dataX = data[:,2:a]
    
    # Creating a model
    model = Sequential()
    model.add(Dense(20, input_dim = dataX.shape[1], activation='elu'))
    model.add(Dense(20, activation='elu'))
    model.add(Dense(20, activation='elu'))
    model.add(Dense(20, activation='elu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='logcosh', optimizer='adam', metrics=['accuracy'])
    result = model.fit(dataX, dataY, validation_split=.33, epochs=150, batch_size=batch, verbose=0,
                       callbacks=[EarlyStopping(patience=10, monitor='val_acc')])
    
    return result

df = pd.read_csv("/home/augusta/dev/autonomio_dev/dev/linear/train.csv")
df = df.drop(df.columns[[3, 8]], axis=1)

# Cleaning the dataset
s = "female"
df.Sex = df.Sex == s
df.Sex = df.Sex.astype(int)

mapping = [('C','0'), ('S','1'), ('Q','2')]
for k,v in mapping:
    df.Embarked = df.Embarked.replace(k,v)

data = np.array(df)

c = len(df) 
b = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']
d = len(b)

for i in range(c):
    a = data[i, 8] != data[i, 8]
    if a == True:
        e = 0
        e = str(e)
        data[i, 8] = e
    else:
        for j in range(d):
            if b[j] in data[i, 8]:
                e = j + 1
                e = str(e)
                data[i, 8] = e
                             
df.Cabin = data[:,8]

df = df.dropna()
data = np.array(df)
data = data.astype(int)

batch = range(4, 50, 1)

def read_file(path):
    l = []
    f = open(path, 'r')
    temp = f.read().split(', ')
    
    for i in range(len(temp) - 1):
        l.append(float(temp[i]))
    f.close()
    
    return l

def safe_file(path, l):
    f = open(path, 'w')
    for x in l:
        f.write(str(x) + ', ')
    f.close()
    
k = 10 * len(batch)
done = 1
a = len(batch) * 10
epoch_time = 0

for i in range(10):
    l = []
    best = []
    best_batch = []
    worst = []
    worst_batch = []
    for b in range(10):
        best.append(0)
        worst.append(0)
        best_batch.append(0)
        worst_batch.append(0)
    
    for x in batch:
        l2 = []
        start_time = dt.datetime.now()
        for y in range(5):
            history = train(data, x)
            l2.append(history.history['val_acc'][-10].mean())
            # Calculating and printing the estimated finish
            if y == 4:
                train_time = dt.datetime.now()
                epoch_time += (long(train_time.strftime('%s')) - long(start_time.strftime('%s'))) / done
                finish_time = dt.datetime.fromtimestamp((epoch_time * k) + long(start_time.strftime('%s')))
                finish_time = finish_time.strftime('%H:%M %d %B')
                print('Estimated finish: ' + finish_time)
                k -= 1
        acc = sum(l2) / 5
        l.append(acc)
        
        # Storing the best and the worst accuracies and the number of batch size roccesponding to that
        if max(worst) > acc and 0 not in worst:
            index = worst.index(max(worst))
            worst[index] = acc
            worst_batch[index] = x
        
        if min(best) < acc and 0 not in best:
            index = best.index(min(best))
            best[index] = acc
            best_batch[index] = x
        
        print('%d out of %d \n' % (done, a))
        
        done += 1
    
    # Saving the file for each try
    path1 = '/home/augusta/dev/autonomio_dev/dev/linear/best_batch' + str(i + 1) + '.txt'
    path2 = '/home/augusta/dev/autonomio_dev/dev/linear/batch_acc_list' + str(i + 1) + '.txt'
    path3 = '/home/augusta/dev/autonomio_dev/dev/linear/worst_acc' + str(i + 1) + '.txt'
    path4 = '/home/augusta/dev/autonomio_dev/dev/linear/best_acc' + str(i + 1) + '.txt'
    path5 = '/home/augusta/dev/autonomio_dev/dev/linear/worst_batch' + str(i + 1) + '.txt'
    
    safe_file(path1, best_batch)
    safe_file(path2, l)
    safe_file(path3, worst)
    safe_file(path4, best)
    safe_file(path5, worst_batch)
