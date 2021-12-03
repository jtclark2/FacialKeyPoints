## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # Initialization: I'm pretty sure glorot and xavier were both authors on that initialization paper
        # so this is glorot init, I think. Glorot uniform is noted in the FKP paper as well (Naimish Agarwal, et al)
        
        # consider adding batch norm throughout???
        dropout_default = 0.2

        num_channels_1 = 16
        self.conv1 = nn.Conv2d(1, num_channels_1, 3) # 224 --> 222
        I.xavier_uniform_(self.conv1.weight)
        self.pool1 = nn.MaxPool2d(2, 2) # 222 --> 111
        self.drop1 = nn.Dropout2d(p=dropout_default)
        
        num_channels_2 = 32
        self.conv2 = nn.Conv2d(num_channels_1, num_channels_2, 3) # 111 --> 109
        I.xavier_uniform_(self.conv2.weight)
        self.pool2 = nn.MaxPool2d(2, 2) # 109 --> 54
        self.drop2 = nn.Dropout2d(p=dropout_default)
        
        num_channels_3 = 64
        self.conv3 = nn.Conv2d(num_channels_2, num_channels_3, 3) # 54 --> 52
        I.xavier_uniform_(self.conv3.weight)
        self.pool3 = nn.MaxPool2d(2, 2) # 52 --> 26
        self.drop3 = nn.Dropout2d(p=dropout_default)
        
        num_channels_4 = 128
        self.conv4 = nn.Conv2d(num_channels_3, num_channels_4, 3) # 26 --> 24
        I.xavier_uniform_(self.conv4.weight, gain=1.0)
        self.pool4 = nn.MaxPool2d(2, 2) # 24 --> 12
        self.drop4 = nn.Dropout2d(p=dropout_default)
        
        num_channels_5 = 256
        self.conv5 = nn.Conv2d(num_channels_4, num_channels_5, 3) # 12 --> 10
        I.xavier_uniform_(self.conv5.weight)
        self.pool5 = nn.MaxPool2d(2, 2) # 10 --> 5
        self.drop5 = nn.Dropout2d(p=dropout_default)
                
        # unroll
        num_dense_nodes_1 = 1000
        self.dense1 = nn.Linear(5*5*num_channels_5, num_dense_nodes_1) # 5*5*256 = 6400
        I.xavier_uniform_(self.dense1.weight)
        # chose 1000 to follow Naimish Agarwal, et al. 
        # on the higher end of what I'd choose - it doubles my total model params
        # it does improve the model a bit...enough to merit the increase? We'll say yes for now.
        #
        num_outputs = 136
        self.dense2 = nn.Linear(num_dense_nodes_1, num_outputs) # 136 outputs
        I.xavier_uniform_(self.dense2.weight)
        
        
    def forward(self, x):
        """
        Args:
            x (pytorch Tensor): Assume incoming image is shape (224,224,1)
        return (pytorch Tensor): List of 136 coordinates (x1,y1,x2,y1,...x68,y68)
            
        """
        
        x = self.drop1(self.pool1(F.relu(self.conv1(x))))
        x = self.drop2(self.pool2(F.relu(self.conv2(x))))
        x = self.drop3(self.pool3(F.relu(self.conv3(x))))
        x = self.drop4(self.pool4(F.relu(self.conv4(x))))
        x = self.drop5(self.pool5(F.relu(self.conv5(x))))
        
        x = x.view(x.size(0), -1) # flatten/unroll
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        # no output activation...so our loss function should just be SME error
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
