
# coding: utf-8

# In[1]:

#Long Short Term Neural Net through Nervana's Neon package


# In[2]:

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# In[3]:

features = pd.read_csv("~/Documents/Walmart Data/features.csv")
features.head()


# In[4]:

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
true_train = true_train.drop(['IsHoliday_y', 'Type'], 1)
true_train = true_train.fillna(true_train.median())
true_train.head()


# In[11]:

true_train.shape


# In[12]:

true_train['IsHoliday_x'] = true_train['IsHoliday_x'].astype(int) #Convert IsHolidayX to a boolean
true_train2 = true_train.sort_values('Date')
true_train2 = true_train2.set_index(true_train2['Date'])
true_train2 = true_train2.drop('Date',1)
true_train2.head()


# In[13]:

def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# In[15]:

values = true_train2.values

values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = timeseries_to_supervised(scaled, 1)


# In[16]:

#reframed.drop(reframed.columns[[14,15,17,18,19,20,21,22,23,24,25,26,27]], axis=1, inplace=True)
#msk = np.random.rand(len(reframed)) < 0.985
n = 415000
train = reframed.iloc[range(n),:]
test = reframed.iloc[range(n, reframed.shape[0]),:]
train_y = scaler.inverse_transform(scaled)[range(n),2:3]
test_y = scaler.inverse_transform(scaled)[range(n, reframed.shape[0]),2:3]
train.drop(train.columns[[2,16]], axis = 1, inplace = True)
test.drop(test.columns[[2,16]], axis = 1, inplace = True)


# In[17]:

test_y


# In[18]:

from neon.data import ArrayIterator
from neon.backends import gen_backend
gen_backend(backend='cpu', batch_size=128)


# In[19]:

#make_onehot makes all our y_labels suitable for classification
train_neon = ArrayIterator(X=train, y=train_y, make_onehot=False)
test_neon = ArrayIterator(X = test, y = test_y, make_onehot = False)


# In[20]:

from neon.initializers import Uniform
init_uniform = Uniform()


# In[21]:

from neon.layers import LSTM, Affine, Dropout, LookupTable, RecurrentLast
from neon.transforms import Logistic, Tanh, Identity, MeanSquared
from neon.models import Model
layers = [
    LSTM(output_size=128, init=init_uniform, activation=Identity(),
         gate_activation=Identity(), reset_cells=True),
    RecurrentLast(),
    Affine(1, init_uniform, bias=init_uniform, activation=Identity())]

model = Model(layers = layers)


# In[22]:

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


# In[23]:

from sklearn.metrics import mean_squared_error
import math


# In[24]:

train_output = model.get_outputs(test_neon)
rmse = math.sqrt(mean_squared_error(train_output, test_y))
nrmse = rmse/np.mean(values)
print('Test RMSE %.3f' % rmse)
print('Test NRMSE % .3f' % nrmse)


# In[25]:

train_output = model.get_outputs(train_neon)
rmse = math.sqrt(mean_squared_error(train_output, train_y))
nrmse = rmse/np.mean(values)
print('Test RMSE %.3f' % rmse)
print('Test NRMSE % .3f' % nrmse)


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



