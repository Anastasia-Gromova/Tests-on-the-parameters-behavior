import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense
import math
import datetime as dt

import matplotlib
import matplotlib.pyplot as plt

plt.style.use('bmh')

def train(data, batch):
    
    # Separating output and input
    a = data.shape[1]-1
    dataY = data[:,1]
    dataX = data[:,2:a]
    
    # Creating the mlp model
    model = Sequential()
    model.add(Dense(20, input_dim = dataX.shape[1], activation='elu'))
    model.add(Dense(20, activation='elu'))
    model.add(Dense(20, activation='elu'))
    model.add(Dense(20, activation='elu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='logcosh', optimizer='adam', metrics=['accuracy'])
    result = model.fit(dataX, dataY, validation_split=.33, epochs=80, batch_size=batch, verbose=0,
                       callbacks=[EarlyStopping(patience=15, monitor='val_acc')])
    
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

# Filling the list of batches (up to 512)
batch2 = [1, 10, 25]
n = 2
for i in range(5, 10):
    prev = batch2[n]

    third = pow(2, i)
    first = prev + (third - prev) / 3
    second = prev + (third - prev) * 2 / 3

    batch2.append(first)
    batch2.append(second)
    batch2.append(third)

    n += 3
    batch2 = [1, 10, 25]
    
l = []
done = 1
k = len(batch2)

# Training the model
for x in batch2:
    
    l2 = []

    for y in range(3):
        history = train(data, x)
        l2.append(history.history['val_acc'][-10].mean())

    acc = sum(l2) / 3
    l.append(acc)
    print ("done " + str(done) + " out of " + str(k))
    done += 1
    
# Showing the plot with the results
plt.plot(batch2, l)
plt.ylim(0, 1)
plt.show()
