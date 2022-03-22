from importlib import reload
import utils.package_imports
reload(utils.package_imports)
from utils.package_imports import *

class LSTM(nn.Module):
    '''
    Pytorch LSTM model class
    Functions:
        init: initializes model based on given parameters
        forward: forward step through model
    '''
    def __init__(self, input_size, hidden_size1, num_layers1, hidden_size2, num_layers2, num_classes, dropout, bidirectional1, bidirectional2, pre_out, single_layer, linear_size, device, use_layernorm):
        super(LSTM, self).__init__()
        self.hidden_size1 = hidden_size1
        self.num_layers1 = num_layers1
        self.input_size = input_size
        self.hidden_size2 = hidden_size2
        self.num_layers2 = num_layers2
        self.num_classes = num_classes
        self.dropout = dropout
        self.bidirectional1 = bidirectional1
        self.bidirectional2 = bidirectional2
        self.pre_out = pre_out
        self.single_lstm = single_layer
        self.linear_size = linear_size
        self.use_layernorm = use_layernorm 
        self.batch_size = 100
        self.device = device
        
        #self.bi_mult1 = 1
        #self.bi_mult2 = 1
        
        #Try splitting lstm into multiple with different hidden sizes 
        
        if (self.bidirectional1):
            self.bi_mult1 = 2
        else:
            self.bi_mult1 = 1      

        if (self.bidirectional2):
            self.bi_mult2 = 2
        else:
            self.bi_mult2 = 1
        
        if (self.single_lstm):
            hidden_size = self.hidden_size1
        else:
            hidden_size = self.hidden_size2

        self.h0, self.c0, self.h02, self.c02 = self.init_hidden()

        self.prefc = nn.Linear(self.input_size, self.pre_out)
        #Does num_layers actually make multiple lstm layers?
        #self.lstm = nn.LSTM(pre_out, hidden_size1, num_layers1, batch_first=True, bidirectional = bidirectional1) #uncomment for first linear layer

        ##Use these two for two seperate lstm layers
        self.lstm = nn.LSTM(self.input_size, self.hidden_size1, self.num_layers1, batch_first=True, bidirectional = self.bidirectional1) #uncomment for first lstm layer

        #self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional = bidirectional) 
        self.lstm2 = nn.LSTM(self.hidden_size1*self.bi_mult1, self.hidden_size2, self.num_layers2, batch_first=True, bidirectional = self.bidirectional2)

        #Batch_first means shape is (batch, seq, feature)
        self.dropout_layer = nn.Dropout(p=dropout)
        #self.fc = nn.Linear(hidden_size * sequence_length, num_classes)
        #self.fc = nn.Linear(hidden_size*2, 1) replaced with below

        self.fc0 = nn.Linear(hidden_size*self.bi_mult2, self.linear_size)

        self.fc1 = nn.Linear(self.linear_size, 2)

        self.fc = nn.Linear(hidden_size*self.bi_mult2, 2)

        self.layernorm = nn.LayerNorm(hidden_size*self.bi_mult2)


        #print("fc constructor: ",hidden_size2*bi_mult2)
        #self.fc2 = nn.Linear(sequence_length, 2)
        self.softmax = nn.Softmax(dim=1)
    
    def init_hidden(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        

        h = torch.empty(self.num_layers1*self.bi_mult1, batch_size, self.hidden_size1)
        c = torch.empty(self.num_layers1*self.bi_mult1, batch_size, self.hidden_size1)
        
        h0 = torch.nn.init.xavier_normal_(h, gain=1.0).to(self.device)
        c0 = torch.nn.init.xavier_normal_(c, gain=1.0).to(self.device)
        
        #h0 = torch.nn.init.kaiming_uniform_(h, a=0, mode='fan_in', nonlinearity='leaky_relu').to(device)
        #c0 = torch.nn.init.kaiming_uniform_(c, a=0, mode='fan_in', nonlinearity='leaky_relu').to(device)

        h2 = torch.empty(self.num_layers2*self.bi_mult2, batch_size, self.hidden_size2)
        c2 = torch.empty(self.num_layers2*self.bi_mult2, batch_size, self.hidden_size2)
        
        h02 = torch.nn.init.xavier_normal_(h2, gain=1.0).to(self.device)
        c02 = torch.nn.init.xavier_normal_(c2, gain=1.0).to(self.device)
        
        #h02 = torch.nn.init.kaiming_uniform_(h2, a=0, mode='fan_in', nonlinearity='leaky_relu').to(device)
        #c02 = torch.nn.init.kaiming_uniform_(c2, a=0, mode='fan_in', nonlinearity='leaky_relu').to(device)
        return h0, c0, h02, c02



    def forward(self, x):
        #print(x.shape)
        #print("x.size(0): ", x.size(0))
        #print(x)
        #print(x.type())
        #print(x[0].type())
        #print(x[0][0].type())


        
        # Set initial hidden and cell states
        #Use xavier initializtion instead of zeros
        #Or something else fancier: kaiming he
        
        
        #Old initialization:
        '''
        h = torch.empty(self.num_layers1*bi_mult1, x.size(0), self.hidden_size1)
        c = torch.empty(self.num_layers1*bi_mult1, x.size(0), self.hidden_size1)
        
        h0 = torch.nn.init.xavier_normal_(h, gain=1.0).to(device)
        c0 = torch.nn.init.xavier_normal_(c, gain=1.0).to(device)
        
        #h0 = torch.nn.init.kaiming_uniform_(h, a=0, mode='fan_in', nonlinearity='leaky_relu').to(device)
        #c0 = torch.nn.init.kaiming_uniform_(c, a=0, mode='fan_in', nonlinearity='leaky_relu').to(device)

        h2 = torch.empty(self.num_layers2*bi_mult2, x.size(0), self.hidden_size2)
        c2 = torch.empty(self.num_layers2*bi_mult2, x.size(0), self.hidden_size2)
        
        h02 = torch.nn.init.xavier_normal_(h2, gain=1.0).to(device)
        c02 = torch.nn.init.xavier_normal_(c2, gain=1.0).to(device)
        
        #h02 = torch.nn.init.kaiming_uniform_(h2, a=0, mode='fan_in', nonlinearity='leaky_relu').to(device)
        #c02 = torch.nn.init.kaiming_uniform_(c2, a=0, mode='fan_in', nonlinearity='leaky_relu').to(device)
        '''
        
        #Initialize initial LSTM states
        h0 = self.h0
        c0 = self.c0
        h02 = self.h02
        c02 = self.c02
        if x.size(0) != self.batch_size:
            h0, c0, h02, c02 = self.init_hidden(x.size(0))
        
        #h02 = torch.nn.init.kaiming_uniform_(h2, a=0, mode='fan_in', nonlinearity='leaky_relu').to(device)
        #c02 = torch.nn.init.kaiming_uniform_(c2, a=0, mode='fan_in', nonlinearity='leaky_relu').to(device)

        #linear layer takes features in and then output as different dimention, change LSTM size to match
        #print("x shape: ", x.shape)

        #out = F.relu(self.prefc(x)) #Uncomment for first linear layer
        #out = (self.prefc(x))
        #print("prefc shape: ", out.shape)

        # Forward propagate LSTM
        #, _ = self.lstm(out, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size) #uncomment for first linear layer
        out, _ = self.lstm(x, (h0, c0))    #uncomment for first lstm layer
        #print("lstm1 out: ", out.shape)
        if (not self.single_lstm):
            out, _ = self.lstm2(out, (h02, c02))
        #print("lstm2 out: ", out.shape)
        #out = out.reshape(out.shape[0], -1)
        #print(torch.isnan(out).any())
        #print("lstm out shape: ", out.shape)
        out = out[:,-1,:]
        #print("Last stride: ", out.shape)
        #print("fixed shape: ", out.shape)
        out = self.dropout_layer(out)
        #print("dropout shape: ", out.shape)
        #print(out)
        #print("after dropout: ", out.shape)
        if self.use_layernorm:
            out = self.layernorm(out)

        #print("fc expected size: ", hidden_size2*bi_mult2)
        #out = self.fc(out[:,:-1])
        #out = F.tanh(self.fc0(out))
        if self.linear_size > 1:
            out = F.tanh(self.fc0(out))
            out = self.fc1(out)
        else:
            out = self.fc(out)
        #print("fully connected 1 shape: ", out.shape)
        #print(out)
        
        #Look at flattening instead of squeeze to reduce 2 fc layers to one
        #out = torch.squeeze(out)
        #out = F.relu(self.fc2(out))
        #print("fully connected 2 shpae: ", out.shape)
        
        #out = self.softmax(out)
        #print("softmax out shape: ", out.shape)
        #print(out)
        
        

        #check: is lstm output 10x30x2? if so it is all timesteps output, we only want the last one out[:,:,-1]? shape should be 30x2 for last one

        
        return out