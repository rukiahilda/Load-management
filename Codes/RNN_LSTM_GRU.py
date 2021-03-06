
# coding: utf-8



# In[2]:
# # Part 2 Time series forecatsing with RNN

# ### Problem Description
# The problem we are going to look at in this post is the International Airline Passengers prediction problem.
# 
# This is a problem where, given a year and a month, the task is to predict the number of international airline passengers in units of 1,000. The data ranges from January 1949 to December 1960, or 12 years, with 144 observations.
# 
# The dataset is available for free from the [DataMarket webpage](https://datamarket.com/data/set/22u3/international-airline-passengers-monthly-totals-in-thousands-jan-49-dec-60#!ds=22u3&display=line) as a CSV download with the filename “international-airline-passengers.csv“.
# 
# 

# In[12]:

import torch
import torch.nn as nn
from torch import cuda, nn, optim
from torch import Tensor
from torch.nn import functional as F
from torch.autograd import Variable






import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(7)


# ### Load  and process data

# In[13]:


data = pd.read_csv('airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)


# In[14]:


data.head()


# The dataset contains n = 41266 minutes of data ranging from April to August 2017 on 500 stocks as well as the total S&P 500 index price. Index and stocks are arranged in wide format.

# A quick look at the S&P time series 

# In[15]:


### visualize data
plt.plot(data);


# You can see an upward trend in the dataset over time.

# #### Process the data
# Normalizing the data using minmax scaler. Neural network architectures benefit from scaling the inputs (sometimes also the output). Why? Because most common activation functions of the network’s neurons such as tanh or sigmoid are defined on the [-1, 1] or [0, 1] interval respectively.

# In[16]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(data.values.astype('float64'))

plt.plot(X)


# #### Train Test data split
# A simple method that we can use is to split the ordered dataset into train and test datasets. The code below calculates the index of the split point and separates the data into the training datasets with 70% of the observations that we can use to train our model, leaving the remaining 30% for testing the model.

# In[17]:


# split into train and test sets
train_size = int(len(X) * 0.7)
test_size = len(X) - train_size
train, test = X[0:train_size,:], X[train_size:len(X),:]
print ("train_data_size: "+str(len(train)), " test_data_size: "+str(len(test)))


# #### Reshape  data into $X={x_t}$ and $Y=x_{t+1}$ using a fixing window length (loop_back)

# In[21]:


def create_window(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY).reshape(-1,1)


# In[22]:


look_back = 1
trainX, trainY = create_window(train, look_back)
testX, testY = create_window(test, look_back)


# ### Define data loader

# In[23]:


from torch.utils.data import Dataset, DataLoader


# In[24]:


class getDataset(Dataset):
    
    def __init__(self, X, y):
        
        self.len = X.shape[0]
        # reshape input to be [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1],1))
        self.x = X
        self.y = y
        print("Size of data: input {0}, label {1}".format(X.shape, y.shape))
        
    def __getitem__(self, index):
        inputs, targets = self.x[index], self.y[index]
        inputs, targets = torch.tensor(inputs).float(), torch.tensor(targets).float()
        
        return inputs, targets
    
    def __len__(self):
        return self.len


# In[25]:


train_set = getDataset(trainX,trainY)
test_set = getDataset(testX,testY)
batch_size = 10
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)


# ### Define Model
# 
# Let define a 2 RNN layers network, with 16 RNN blocks or neurons and an output layer that makes a single value prediction. 

# In[26]:


rnn = RNNmodel(in_dim=1, hidden_size=64, layer_size=2, out_dim=1)
print(rnn)


# ### Define a Loss function and Optimizer

# In[27]:


learning_rate = 0.001
criterion = nn.MSELoss()
optimizer =  torch.optim.Adam(rnn.parameters(), lr=learning_rate)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# #### Define training loop

# In[28]:


def train(model, optimizer, criterion,  device, num_epochs):
    
    total_loss = []
    print("Start training")
    model.to(device)
    criterion.to(device)
    
    for epoch in range(num_epochs):
        
        training_loss = []
        model.train()
        
        for i, (inputs, targets) in enumerate(train_loader):
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
        
            pred = model(inputs)
        
            # Calculate Loss: 
            loss = criterion(pred, targets)
            training_loss.append(loss.item())
            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()
    
        total_loss.append(np.mean(training_loss))
        if epoch% 50==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]  Loss: {:.6f}'.format(
                epoch+1, i * len(inputs), len(train_loader.dataset),
                100. * i / len(train_loader), np.mean(training_loss)))
        
    return total_loss


# In[29]:


rnn_loss = train(rnn, optimizer,criterion,  device, 1000)


# In[30]:


plt.plot(rnn_loss, label="rate={}".format(learning_rate))
plt.title("Training Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()


# ### Let se how the model perform

# In[31]:


def predict(model, device, inputs):
    model.to(device)
    model.eval()
    
    inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1],1))
    inputs = torch.tensor(inputs).float()
    predict = model(inputs)
    
    return predict.detach().numpy()


# In[32]:


pred_rnn = predict(rnn, device, testX)


# ### Invert prediction to actual values

# In[34]:


y_pred_rnn = scaler.inverse_transform(pred_rnn)
y_true = scaler.inverse_transform(testY)


# In[35]:


plt.plot(y_true, label="Actual passenger number")
plt.plot(y_pred_rnn, label="Predicted passenger number")
plt.legend()


# ### Use LSTM instead

# In[36]:


lstm =  LSTMmodel(in_dim=1, hidden_size=64, layer_size=2, out_dim=1)
optimizer =  torch.optim.Adam(lstm.parameters(), lr=learning_rate)
total_loss = train(lstm, optimizer,criterion,  device, 900)


# In[37]:


plt.plot(rnn_loss, label="RNN with rate={}".format(learning_rate))
plt.plot(total_loss, label="LSTM with rate={}".format(learning_rate))
plt.title("Training Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()


# In[38]:


pred_lstm = predict(lstm, device, testX)
y_pred_lstm = scaler.inverse_transform(pred_lstm)


# In[39]:


plt.plot(y_true, label="Actual passenger number")
plt.plot(y_pred_rnn, label="RNN Predicted passenger number")
plt.plot(y_pred_lstm, label="LSTM Predicted passenger number")
plt.legend()


# In[40]:


### Let check metric score
from sklearn.metrics import mean_squared_error


# In[41]:


print('Test Score:RNN {}, LSTM {}'.format(mean_squared_error(y_true, y_pred_rnn), mean_squared_error(y_true, y_pred_lstm)))

