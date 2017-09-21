
# coding: utf-8

# In[29]:

#Long Short Term Neural Net using Python


# In[30]:

import pandas as pd
import numpy as np
import scipy as sp
import keras
from ggplot import *
from matplotlib import pyplot


# In[31]:

features = pd.read_csv("~/Documents/Walmart Data/features.csv")
features.head()


# In[32]:

features.shape


# In[33]:

pre_train = pd.read_csv("~/Documents/Walmart Data/train 2.csv")
pre_train.head()


# In[34]:

pre_train.shape


# In[35]:

stores = pd.read_csv("~/Documents/Walmart Data/stores.csv")
stores.head()


# In[36]:

stores.shape


# In[37]:

true_train = pd.merge(pd.merge(pre_train, features, how = 'left',
                               left_on = ['Store', 'Date'], right_on = ['Store', 'Date']), 
                      stores, left_on = ['Store'], right_on = ['Store'])
true_train['Date'] = pd.to_datetime(true_train['Date'])
true_train = true_train.drop(['IsHoliday_y', 'Type'], 1)
true_train = true_train.fillna(true_train.median())
true_train.head()


# In[38]:

true_train.shape


# In[39]:

#Exploratory Analysis


# In[40]:

ggplot(aes(x = 'Store', y = 'Weekly_Sales', color = 'factor(Store)'), true_train) + geom_point() +theme_bw()


# In[41]:

ggplot(aes(x = 'Date', y = 'Weekly_Sales', colour = 'factor(Store)'), true_train) + geom_point() +scale_x_date(format = '%b-%Y') +theme_bw()


# In[42]:

ggplot(aes('Fuel_Price', "Weekly_Sales", colour = 'factor(Store)'), true_train) + geom_point() +theme_bw()


# In[43]:

ggplot(aes('IsHoliday_x', "Weekly_Sales"), true_train) + geom_boxplot() +theme_bw()


# In[44]:

#LTSM Neural Net


# In[45]:

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot


# In[46]:

true_train['IsHoliday_x'] = true_train['IsHoliday_x'].astype(int) #Convert IsHolidayX to a boolean
true_train2 = true_train.sort_values('Date')
true_train2 = true_train2.set_index(true_train2['Date'])
true_train2 = true_train2.drop('Date',1)
true_train2.head()


# In[47]:

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Returns a time series into a dataset applicable for supervised learning
    
    Arguments:
    data -- the timeseries data to be converted into supervised learning data
    n_in -- determines input sequence n. Default is n = 1
    n_out -- determines forecast sequence n. Default is n = 1
    dropnan -- True/False if the user wishes to eliminate all row values with a NaN in them. 
    Default is True.
    
    Returns:
    agg - the supervised learning dataset
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# In[48]:

#true_train2 = true_train2.drop('Weekly_Sales', 1)

msk = np.random.rand(len(true_train2)) < 0.8

train = true_train2[msk]
test = true_train2[~msk]
train_y = train['Weekly_Sales']
test_y = test['Weekly_Sales']
train = train.drop('Weekly_Sales',1)
test = test.drop('Weekly_Sales',1)

values = train.values

values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict

# reshape input to be 3D [samples, timesteps, features]
train_X = values.reshape((train.shape[0], 1, train.shape[1]))
#test_X = true_test2.reshape((true_test.shape[0], 1, true_test.shape[1]))


# In[49]:

values = test.values

values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict

# reshape input to be 3D [samples, timesteps, features]
test_X = values.reshape((test.shape[0], 1, test.shape[1]))


# In[50]:

model = Sequential()
#return_sequences = True allows us to build multi-layer NNs
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences = True))
model.add(LSTM(25, return_sequences = True))
model.add(LSTM(10))
#Dense determines how many outputs come from our NN
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=60, batch_size=64, 
                    validation_data=(test_X, test_y), verbose=2, shuffle=False)


# In[ ]:

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
 
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


# In[ ]:

#Setting Up WalMart Test Data for Predictions


# In[ ]:

pre_test = pd.read_csv("~/Documents/Walmart Data/test 2.csv")
pre_test.head()


# In[ ]:

pre_test.shape


# In[ ]:

true_test = pd.merge(pd.merge(pre_test, features, how = 'left',
                              left_on = ['Store', 'Date'], right_on = ['Store', 'Date']),
                     stores, left_on = ['Store'], right_on = ['Store'])
true_test['Date'] = pd.to_datetime(true_test['Date'])
true_test = true_test.drop(['IsHoliday_y', 'Type'], 1)
true_test = true_test.fillna(true_test.median())
true_test.head()


# In[ ]:

true_test.shape


# In[ ]:

true_test['IsHoliday_x'] = true_test['IsHoliday_x'].astype(int)
true_test2 = true_test.sort_values('Date')
true_test2 = true_test2.set_index(true_test2['Date'])
true_test2 = true_test2.drop('Date',1)
true_test2.head()

