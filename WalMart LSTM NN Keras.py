
# coding: utf-8

# In[2]:

#Long Short Term Neural Net using Python


# In[3]:

import pandas as pd
import numpy as np
import keras
from ggplot import *
from matplotlib import pyplot
import math


# In[4]:

features = pd.read_csv("~/Documents/Walmart Data/features.csv")
features.head()


# In[5]:

features.shape


# In[6]:

pre_train = pd.read_csv("~/Documents/Walmart Data/train 2.csv")
pre_train.head()


# In[7]:

pre_train.shape


# In[8]:

stores = pd.read_csv("~/Documents/Walmart Data/stores.csv")
stores.head()


# In[9]:

stores.shape


# In[10]:

true_train = pd.merge(pd.merge(pre_train, features, how = 'left',
                               left_on = ['Store', 'Date'], right_on = ['Store', 'Date']), 
                      stores, left_on = ['Store'], right_on = ['Store'])
true_train['Date'] = pd.to_datetime(true_train['Date'])
true_train = true_train.drop(['IsHoliday_y', 'Type', 'MarkDown1', 
                             'MarkDown2', 'MarkDown3', 'MarkDown4',
                             'MarkDown5'], 1)
true_train = true_train.fillna(true_train.median())
true_train.head()


# In[11]:

true_train.shape


# In[12]:

#Exploratory Analysis


# In[13]:

ggplot(aes(x = 'Store', y = 'Weekly_Sales', color = 'factor(Store)'), true_train) + geom_point() +theme_bw()


# In[14]:

ggplot(aes(x = 'Date', y = 'Weekly_Sales', colour = 'factor(Store)'), true_train) + geom_point() +scale_x_date(format = '%b-%Y') +theme_bw()


# In[15]:

ggplot(aes('Fuel_Price', "Weekly_Sales", colour = 'factor(Store)'), true_train) + geom_point() +theme_bw()


# In[16]:

ggplot(aes('IsHoliday_x', "Weekly_Sales"), true_train) + geom_boxplot() +theme_bw()


# In[17]:

#LTSM Neural Net


# In[18]:

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot


# In[19]:

true_train['IsHoliday_x'] = true_train['IsHoliday_x'].astype(int) #Convert IsHolidayX to a boolean
true_train2 = true_train.sort_values('Date')
true_train2 = true_train2.set_index(true_train2['Date'])
true_train2 = true_train2.drop('Date',1)
true_train2.head()


# In[20]:

def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# In[21]:

values = true_train2.values

values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = timeseries_to_supervised(scaled, 1)


# In[23]:

reframed.head()


# In[26]:

#reframed.drop(reframed.columns[[range(1:13)]], axis=1, inplace=True)
#msk = np.random.rand(len(reframed)) < 0.985
n = 415000
train = reframed.iloc[range(n),:]
test = reframed.iloc[range(n, reframed.shape[0]),:]
train_y = scaler.inverse_transform(scaled)[range(n),2]
test_y = scaler.inverse_transform(scaled)[range(n, reframed.shape[0]),2]
train.drop(train[2], axis = 1, inplace = True)
test.drop(test[2], axis = 1, inplace = True)

train.head()


# In[27]:

test.head()


# In[28]:

train_y


# In[29]:

test_y


# In[30]:

train_X = train.values.reshape((train.shape[0], 1, train.shape[1]))
test_X = test.values.reshape((test.shape[0], 1, test.shape[1]))


# In[ ]:

#Multiple Hidden Layers
#model = Sequential()
#return_sequences = True allows us to build multi-layer NNs
#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences = True))
#model.add(LSTM(25, return_sequences = True))
#model.add(LSTM(12))
#Dense determines how many outputs come from our NN
#model.add(Dense(1))

#model.compile(loss='mse', optimizer='adam')
# fit network
#history = model.fit(train_X, train_y, epochs=25, batch_size=64, 
#                    validation_data=(test_X, test_y), verbose=2, shuffle=False)


# In[31]:

model = Sequential()
#return_sequences = True allows us to build multi-layer NNs
model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2])))
#Dense determines how many outputs come from our NN
model.add(Dense(1))
model.compile(loss='mse', optimizer='adagrad')
# fit network
history = model.fit(train_X, train_y, epochs=25, batch_size=64, 
                    validation_data=(test_X, test_y), verbose=2, shuffle=False)


# In[32]:

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# In[33]:

yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
test_y = test_y.reshape((len(test_y), 1))

# calculate RMSE
mse = mean_squared_error(test_y, yhat)
print('Test MSE: %.3f' % mse)


# In[35]:

rmse = math.sqrt(mean_squared_error(test_y, yhat))
#nrmse = rmse/np.ptp(values)
nrmse = rmse/np.mean(values)
print('Test RMSE %.3f' % rmse)
print('Test NRMSE % .3f' % nrmse)


# In[36]:

#Setting Up WalMart Test Data for Predictions


# In[37]:

pre_test = pd.read_csv("~/Documents/Walmart Data/test 2.csv")
pre_test.head()


# In[38]:

pre_test.shape


# In[48]:

true_test = pd.merge(pd.merge(pre_test, features, how = 'left',
                              left_on = ['Store', 'Date'], right_on = ['Store', 'Date']),
                     stores, left_on = ['Store'], right_on = ['Store'])
true_test['Date'] = pd.to_datetime(true_test['Date'])
true_test = true_test.drop(['IsHoliday_y', 'Type', 'MarkDown1', 
                             'MarkDown2', 'MarkDown3', 'MarkDown4',
                             'MarkDown5'], 1)
true_test = true_test.fillna(true_test.median())
true_test.head()


# In[49]:

true_test.shape


# In[50]:

true_test['IsHoliday_x'] = true_test['IsHoliday_x'].astype(int)
true_test2 = true_test.sort_values('Date')
true_test2 = true_test2.set_index(true_test2['Date'])
true_test2 = true_test2.drop('Date',1)
true_test2.head()


# In[51]:

values = true_test2.values

values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = timeseries_to_supervised(scaled, 1)


# In[52]:

final_train = reframed.values.reshape((reframed.shape[0], 1, reframed.shape[1]))
final_train


# In[53]:

model.predict(final_train)
