
# coding: utf-8

# In[79]:

#Long Short Term Neural Net through Nervana's Neon package


# In[80]:

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# In[81]:

features = pd.read_csv("~/Documents/Walmart Data/features.csv")
features.head()


# In[82]:

features.shape


# In[83]:

pre_train = pd.read_csv("~/Documents/Walmart Data/train 2.csv")
pre_train.head()


# In[84]:

pre_train.shape


# In[85]:

stores = pd.read_csv("~/Documents/Walmart Data/stores.csv")
stores.head()


# In[86]:

stores.shape


# In[87]:

#Using essentially a SQL left join to merge the train, features and store data together
true_train = pd.merge(pd.merge(pre_train, features, how = 'left',
                               left_on = ['Store', 'Date'], right_on = ['Store', 'Date']), 
                      stores, left_on = ['Store'], right_on = ['Store'])
#Convert the date variable from a string to datetime
true_train['Date'] = pd.to_datetime(true_train['Date'])
#Removed columns that were either redundant, a mixture of string and ints or
#had too many NAs
true_train = true_train.drop(['IsHoliday_y', 'Type', 'MarkDown1', 
                             'MarkDown2', 'MarkDown3', 'MarkDown4',
                             'MarkDown5'], 1)
#Any NAs or NaNs became the column median
true_train = true_train.fillna(true_train.median())
true_train.head()


# In[88]:

true_train.shape


# In[89]:

true_train['IsHoliday_x'] = true_train['IsHoliday_x'].astype(int) #Convert IsHolidayX to a boolean
true_train2 = true_train.sort_values('Date')
true_train2 = true_train2.set_index(true_train2['Date'])
true_train2 = true_train2.drop('Date',1)
true_train2.head()


# In[90]:

def timeseries_to_supervised(data, lag=1):
    """
    This function converts a pandas dataframe/Series/numpy array organized 
    by time into one that is suitable for a supervised learning problem.
    
    Parameters
    ----------
    data: pandas DataFrame or pandas Series or numpy Array
        Time Series data
    lag: integer
        The amount of "time lag" we wish our model to have
        
    Returns
    -------
    df: pandas DataFrame
        TimeSeries dataframe that is suitable for supervised learning
    """
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# In[91]:

values = true_train2.values

values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = timeseries_to_supervised(scaled, 1)


# In[96]:

n = 415000
#Since this is a timeseries data, I don't divide the data into random train and test sets
#because we're trying to use information from previous weeks to influence our result
train = reframed.iloc[range(n),:]
test = reframed.iloc[range(n, reframed.shape[0]),:]
#The reason I include 2:3 is because Neon can't deal with one-dimensional arrays with
#no specified height, i.e. x.shape returns (600, ), thus you need to convert the arrays
#into one-dimensional arrays with a specified height
train_y = scaler.inverse_transform(scaled)[range(n),2:3]
test_y = scaler.inverse_transform(scaled)[range(n, reframed.shape[0]),2:3]
#Removes the weekly_sales data and the time-lagged weekly_sales data. 
#This is done only because the actual WalMart test data does not 
#include any weekly_sales to predict from.
train.drop(train[2], axis = 1, inplace = True)
test.drop(test[2], axis = 1, inplace = True)

train.head()


# In[97]:

test_y


# In[98]:

from neon.data import ArrayIterator
from neon.backends import gen_backend
gen_backend(backend='cpu', batch_size=128)


# In[99]:

#make_onehot makes all our y_labels suitable for classification
train_neon = ArrayIterator(X=train, y=train_y, make_onehot=False)
test_neon = ArrayIterator(X = test, y = test_y, make_onehot = False)


# In[100]:

from neon.initializers import Uniform
init_uniform = Uniform()


# In[115]:

from neon.layers import LSTM, Affine, Dropout, LookupTable, RecurrentLast, RecurrentSum
from neon.transforms import Logistic, Tanh, Identity, MeanSquared
from neon.models import Model

#Single Layer
layers = [
    LSTM(output_size=128, init=init_uniform, activation=Identity(),
         gate_activation=Identity(), reset_cells=True),
    RecurrentLast(),
    Affine(1, init_uniform, bias=init_uniform, activation=Identity())]

model = Model(layers = layers)


# In[135]:

#Multiple Layers
layers = [
    LSTM(output_size=64, init=init_uniform, activation=Identity(),
         gate_activation=Identity(), reset_cells=True),
    Affine(nout = 32, init = init_uniform, bias = init_uniform, activation = Identity()),
    Affine(1, init_uniform, bias=init_uniform, activation=Identity())]

model = Model(layers = layers)


# In[131]:

#layers = [
#    LSTM(output_size=64, init=init_uniform, activation=Identity(),
#         gate_activation=Identity(), reset_cells=True),
#    Affine(nout = 32, init = init_uniform, bias = init_uniform, activation = Identity()),
#    Affine(nout = 8, init = init_uniform, bias = init_uniform, activation = Identity()),
#    RecurrentLast(),
#    Affine(1, init_uniform, bias=init_uniform, activation=Identity())]


# In[ ]:

from neon.layers import GeneralizedCost
from neon.optimizers import Adagrad
from neon.callbacks import Callbacks

callbacks = Callbacks(model, eval_set=test_neon, eval_freq = 1)
cost = GeneralizedCost(costfunc=MeanSquared())
optimizer = Adagrad(learning_rate=0.01)

model.fit(train_neon,
          optimizer=optimizer,
          num_epochs=25,
          cost=cost,
          callbacks = callbacks)


# In[127]:

from sklearn.metrics import mean_squared_error
import math


# In[133]:

train_output = model.get_outputs(test_neon)
mse = mean_squared_error(train_output, test_y)
rmse = math.sqrt(mse)
nrmse = rmse/np.mean(values)
nrmse = rmse/np.mean(values)
print('Test MSE %.3f' % mse)
print('Test RMSE %.3f' % rmse)
print('Test NRMSE % .3f' % nrmse)


# In[134]:

train_output = model.get_outputs(train_neon)
mse = mean_squared_error(train_output, train_y)
rmse = math.sqrt(mse)
nrmse = rmse/np.mean(values)
print('Train MSE %.3f' % mse)
print('Train RMSE %.3f' % rmse)
print('Train NRMSE % .3f' % nrmse)
#One reason the train error is higher than the test error is because, since our NN is 
#dealing with so much more dates in the training as opposed to the test data, it had to deal
#with "harder" examples whereas our test data was "easier" to figure out. 


# In[26]:

#Setting Up Data for Predictions


# In[27]:

pre_test = pd.read_csv("~/Documents/Walmart Data/test 2.csv")
pre_test.head()


# In[28]:

pre_test.shape


# In[29]:

true_test = pd.merge(pd.merge(pre_test, features, how = 'left',
                              left_on = ['Store', 'Date'], right_on = ['Store', 'Date']),
                     stores, left_on = ['Store'], right_on = ['Store'])
true_test['Date'] = pd.to_datetime(true_test['Date'])
true_test = true_test.drop(['IsHoliday_y', 'Type'], 1)
true_test = true_test.fillna(true_test.median())
true_test.head()


# In[30]:

true_test.shape


# In[31]:

true_test['IsHoliday_x'] = true_test['IsHoliday_x'].astype(int)
true_test2 = true_test.sort_values('Date')
true_test2 = true_test2.set_index(true_test2['Date'])
true_test2 = true_test2.drop('Date',1)
true_test2.head()


# In[50]:

values = true_test2.values

values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = timeseries_to_supervised(scaled, 1)


# In[51]:

fin_test = ArrayIterator(X = reframed)


# In[71]:

pred = model.get_outputs(fin_test)
pred2 = pred.flat[:]
pred2


# In[73]:

sample = pd.read_csv("~/Documents/Walmart Data/sampleSubmission.csv")
length = np.array(sample.iloc[:,0])


# In[74]:

fin_df = pd.DataFrame({'Id': length, 'Weekly_Sales': pred2})


# In[78]:

np.savetxt('Neon.csv', X = fin_df, delimiter = ',', header = 'Id,Weekly_Sales', comments = '', fmt = '%s')


# In[56]:

reframed.shape


# In[ ]:



