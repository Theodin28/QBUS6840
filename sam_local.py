#!/usr/bin/env python
# coding: utf-8

# # QBUS6840 Assignment 2
# ### RNN LSTM Model
# Sam Curtis

# In[39]:


import pandas as pd
import numpy as np
from tqdm import tqdm_notebook

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Import and View Data

# In[7]:


data = pd.read_csv('C:/Users/Sam/Documents/Github/QBUS6840/UnemploymentRateJan1986-Dec2018.csv')
data['Months']=pd.to_datetime(data['Months'],format='%b-%y')   # set month variable to datetime
data.set_index('Months', inplace=True)
ts = data['Unemployment_Rates']
data.head()


# In[8]:


data.describe()


# In[9]:


from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
plt.figure()
plt.plot(ts)
plt.title('Unemployment Rates from Jan 1986 to Dec 2018')
plt.xlabel('Time')
plt.ylabel('Unemployment Rates');


# ## Data Preprocessing
# Change values to float

# In[11]:


unemployment_data = data['Unemployment_Rates'].values.astype(float)


# ### Train/Test Split
# Set the test window size to 12 considering we will be forecasting 12 months

# In[12]:


test_size = 12
train_data = unemployment_data[:-test_size]
test_data = unemployment_data[-test_size:]

print(len(train_data))
print(len(test_data))


# ### Normalise the data
# We can try with / without normalisation, normally for time series data it is suggested to use normalisation. Typically this has to do with factors such as increasing variability throughout time or trends however this may not be the case in unemployment data... im not sure..
#   
# I have used sklearn minmaxscaler which will be used to normalise data to a (-1, 1) range
#   
# IMPORTANT: normalisation is only applied on the training data and not the test data. If normalisation is applied on the test data, there is a chance that some of the information will be leaked form the training set into the test set.

# In[13]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1,1))
train_data_normalised = scaler.fit_transform(train_data.reshape(-1,1))


# ### Convert data into tensors

# In[18]:


train_data_normalised = torch.FloatTensor(train_data_normalised).view(-1)


# ### Convert training data into sequences with corresponding labels
# We have monthly data so will convert our data into sequences of length 12

# In[20]:


train_window = 12


# Define function called create_inout_sequences. The function will accrept raw input data and will return a list of tuples.
# First tuple contains list of 12 items corresponding to the unemployment rate for the most recent 12 months.
# Second tuple contains one item: the one-step ahead forecast.
#   
# IMPORTANT: This means we will use forecasts to generate the next forecast.

# In[21]:


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq


# In[22]:


train_inout_seq = create_inout_sequences(train_data_normalised, train_window)


# In[23]:


train_inout_seq[:5]


# ## Creating LSTM Model
# ### Model

# In[24]:


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


# ### Criterion

# In[26]:


epochs = 150
l_rate = 0.1     # loss bounces around too much with l_rate = 0.25

model = LSTM()
loss_function = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=l_rate)
print(model)


# ### Train the Model

# In[32]:


for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimiser.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimiser.step()

    print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')


# ## Generate Forecasts
# Initially, test_inputs will contain the last 12 observations of the training data (Jan-Dec '18).
# The model will then forecast Jan '19 and append it to test_inputs.
# The model will then use Feb'18-Jan'19 (which will be a forecasted value) to forecast Feb'19, and so on.
#   
# The for loop will execute 12 times since there are 12 elements in the test set. At the end of the loop, test_inputs will contain 24 items, the last 12 of which will be the rpedicted values for the test set. 

# In[34]:


# Filter out the last 12 obvservations from the TRAINING set    (is not the final 12 observations of the total dataset)
prediction_range = 12

test_inputs = train_data_normalised[-train_window:].tolist()
print(test_inputs)


# #### Execute loop to make forecasts

# In[36]:


model.eval()

for i in range(prediction_range):
    seq = torch.FloatTensor(test_inputs[-train_window:])                # start the loop again taking the last 12 observations (dropping the first one from last time)
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_inputs.append(model(seq).item())                           # append the model forecast to the test_inputs list


# In[37]:


test_inputs[prediction_range:]


# Since the input data was normalised we need to convert it back

# In[40]:


model_forecasts = scaler.inverse_transform(np.array(test_inputs[prediction_range:]).reshape(-1,1))
print(model_forecasts)


# ## Visualise Forecasts

# In[46]:


forecast_dates = data.index[-12:]

rcParams['figure.figsize'] = 15, 6
plt.figure()
plt.plot(ts)
plt.plot(forecast_dates, model_forecasts)
plt.title('Unemployment Rates from Jan 1986 to Dec 2018')
plt.xlabel('Time')
plt.ylabel('Unemployment Rates');


# #### To get a more zoomed in view

# In[47]:


rcParams['figure.figsize'] = 15, 6
plt.figure()
plt.plot(ts[-train_window:])
plt.plot(forecast_dates, model_forecasts)
plt.title('Unemployment Rates from Jan 1986 to Dec 2018')
plt.xlabel('Time')
plt.ylabel('Unemployment Rates');



# In[48]:

# Calc MSE for the test set
print('Forecast MSE = %.3f' %(loss_function(torch.Tensor(data['Unemployment_Rates'][-12:]), torch.Tensor(model_forecasts)[:,-1])))  
