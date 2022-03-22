from importlib import reload
import utils.package_imports
reload(utils.package_imports)
from utils.package_imports import *

class CNN_LSTM(nn.Module):
    '''
    Pytorch CNN LSTM model class
    Functions:
        init: initializes model based on given parameters
        forward: forward step through model
    '''
    def __init__(self, input_size, out_channels, kernel_size, hidden_size1, num_layers1, dropout, bidirectional1, linear_size, device,):
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(21, self.out_channels, kernel_size=self.kernel_size) #Output shape
        self.lstm = nn.LSTM(self.input_size, self.hidden_size1, self.num_layers1, batch_first=True, bidirectional = self.bidirectional1)
        
    
    def forward(self, x):
        out = self.conv1(x)

