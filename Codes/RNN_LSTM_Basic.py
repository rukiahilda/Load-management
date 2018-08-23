
# coding: utf-8

# In[ ]:
# In[1]:

 ## Part 1: Defining RNN in pytorch

# ## RNN
# 
# ```python
#    nn.RNN(input_size,hidden_size, num_layers, nonlinearity, bias, batch_first, dropout, bidirectional)
# ```
# 
# 
# - input_size – The number of expected features in the input x
# - hidden_size – The number of features in the hidden state h
# - num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together to form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results. Default: 1
# - nonlinearity – The non-linearity to use. Can be either ‘tanh’ or ‘relu’. Default: ‘tanh’
# - bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
# - batch_first – If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
# - dropout – If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0
# - bidirectional – If True, becomes a bidirectional RNN. Default: False
# 
# 
# The RNN model expect two inputs: 
# - input data (x) of shape (seq_len, batch, input_size), 
# - initial hidden state (h_0) of shape (num_layers * num_directions, batch, hidden_size)
# 
# It also return two output
# - output of shape (seq_len, batch, num_directions * hidden_size): tensor containing the output features ($h_t$) from the last layer of the RNN, for each $t$. 
# - h_n (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state 


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

# #### Example
# ## GRU
# 
# 
# ```python
#    nn.GRU(input_size,hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
# ```
# 
# - **Input**:  input, h_0
# - **Output**: output, h_n

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


rnn = RNNmodel(3, 64, 2, 1)

out = rnn(x)


out.shape



# In[9]:


#define RNN with 
lstm = nn.LSTM(input_size=3, hidden_size=64, num_layers=2, batch_first=True)
print(lstm)

#define initial hidden state
h_0 =torch.zeros(2, 32, 64)
c_0 =torch.zeros(2, 32, 64)

output, (h_n, c_n) = lstm(x, (h_0, c_0))


# #### LSTM class

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

lstm = LSTMmodel(3, 64, 2, 1)
out = lstm(x)

