
# coding: utf-8

# In[ ]:


#define data of size (batch, seq_len, dim)
x = torch.randn(32, 60, 3)

#define RNN with 
rnn = nn.RNN(input_size=3, hidden_size=64, num_layers=2, batch_first=True)
print(rnn)

#define initial hidden state
h_0 =torch.zeros(2, 32, 64)

#get out and new hidden state
out, hidden = rnn(x, h_0)


# #### RNN model class

# In[4]:


class RNNmodel(nn.Module):
    def __init__(self, in_dim, hidden_size, layer_size, out_dim):
        super(RNNmodel, self).__init__()
        self.hidden_size = hidden_size
        self.layer_size  = layer_size
        self.out_dim = out_dim
        self.rnn = nn.RNN(input_size=in_dim, hidden_size=hidden_size, num_layers=layer_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, out_dim)
        
    def forward(self, x):
        h_0 = torch.zeros(self.layer_size, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h_0)
        out = torch.sigmoid(self.fc(out))
        return out[:,-1,:]


# In[7]:


rnn = RNNmodel(3, 64, 2, 1)


# In[11]:


out = rnn(x)


# In[8]:


out.shape


# ## LSTM
# 
# ```python
#    nn.LSTM(input_size,hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
# ```
# 
# - **Input**:  input, (h_0, c_0)
# - **Output**: output, (h_n, c_n)
# 
# #### Example

# In[9]:


#define RNN with 
lstm = nn.LSTM(input_size=3, hidden_size=64, num_layers=2, batch_first=True)
print(lstm)

#define initial hidden state
h_0 =torch.zeros(2, 32, 64)
c_0 =torch.zeros(2, 32, 64)

output, (h_n, c_n) = lstm(x, (h_0, c_0))


# #### LSTM class

# In[10]:


class LSTMmodel(nn.Module):
    def __init__(self, in_dim, hidden_size, layer_size, out_dim):
        super(LSTMmodel, self).__init__()
        self.hidden_size = hidden_size
        self.layer_size  = layer_size
        self.out_dim = out_dim
        self.rnn = nn.LSTM(input_size=in_dim, hidden_size=hidden_size, num_layers=layer_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, out_dim)
        
    def forward(self, x):
        h_0 = torch.zeros(self.layer_size, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.layer_size, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, (h_0, c_0))
        out = torch.sigmoid(self.fc(out))
        return out[:,-1,:]


# In[11]:


lstm = LSTMmodel(3, 64, 2, 1)
out = lstm(x)

