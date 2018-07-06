import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.autograd import Variable


def MNISTdataLoader(path):
    ##load moving mnist data, data shape = [time steps, batch size, width, height] = [20, batch_size, 64, 64]
    data = np.load(path)
    train = data[:, 0:7000, :, :]
    test = data[:, 7000:10000, :, :]
    return train, test

class MovingMNISTdataset(Dataset):
    ##dataset class for moving MNIST data
    def __init__(self, path):
        self.path = path
        self.train, self.test = MNISTdataLoader(path)

    def __len__(self):
        return len(self.train[0])

    def __getitem__(self, indx, mode = "train"):
        ##getitem method
        if mode == "train":
            self.trainsample_ = self.train[:, 10*indx:10*(indx+1), :, :]
            self.sample_ = self.trainsample_

        if mode == "test":
            self.testsample_ = self.test[:, 10*indx:10*(index+1), :, :]
            self.sample_ = self.trainsample_

        self.sample = torch.from_numpy(np.expand_dims(self.sample_, axis = 2)).float()
        return self.sample

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class CLSTM_cell(nn.Module):
    """ConvLSTMCell
    """
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CLSTM_cell, self).__init__()

        self.shape = shape ##H, W
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1)/2
        self.conv = nn.Conv2d(self.input_channels + self.num_features, 4*self.num_features, self.filter_size, 1, self.padding)

    def forward(self, input, hidden_state):
        hx, cx = hidden_state
        combined = torch.cat((input, hx), 1)
        gates = self.conv(combined)

        ingate, forgetgate, cellgate, outgate = torch.split(gates, self.num_features, dim=1)
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate*cx) + (ingate*cellgate)
        hy = outgate * F.tanh(cy)

        return hy, cy

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.num_features, self.shape[0], self.shape[1])).cuda(), 
                Variable(torch.zeros(batch_size, self.num_features, self.shape[0], self.shape[1])).cuda())

class CLSTM(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size
      
    """
    def __init__(self, shape, input_chans, filter_size, num_features,num_layers):
        super(CLSTM, self).__init__()
        
        self.shape = shape#H,W
        self.input_chans=input_chans
        self.filter_size=filter_size
        self.num_features = num_features
        self.num_layers=num_layers
        cell_list=[]
        cell_list.append(CLSTM_cell(self.shape, self.input_chans, self.filter_size, self.num_features).cuda())#the first
        #one has a different number of input channels
        
        for idcell in xrange(1,self.num_layers):
            cell_list.append(CLSTM_cell(self.shape, self.num_features, self.filter_size, self.num_features).cuda())
        self.cell_list=nn.ModuleList(cell_list)      

    
    def forward(self, input, hidden_state):
        """
        args:
            hidden_state:list of tuples, one for every layer, each tuple should be hidden_layer_i,c_layer_i
            input is the tensor of shape seq_len,Batch,Chans,H,W
        """

        #current_input = input.transpose(0, 1)#now is seq_len,B,C,H,W
        current_input=input
        next_hidden=[]#hidden states(h and c)
        seq_len=current_input.size(0)

        
        for idlayer in xrange(self.num_layers):#loop for every layer

            hidden_c=hidden_state[idlayer]#hidden and c are images with several channels
            all_output = []
            output_inner = []

            for t in xrange(seq_len):#loop for every step
                hidden_c=self.cell_list[idlayer](current_input[t, :, :, :, :],hidden_c)#cell_list is a list with different conv_lstms 1 for every layer
                output_inner.append(hidden_c[0])

            next_hidden.append(hidden_c)
            current_input = torch.cat(output_inner, 0).view(current_input.size(0), *output_inner[0].size())#seq_len,B,chans,H,W


        return next_hidden, current_input

    def init_hidden(self,batch_size):
        init_states=[]#this is a list of tuples
        for i in xrange(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

class MNISTDecoder(nn.Module):
    """
    Decoder for MNIST
    """
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(MNISTDecoder, self).__init__()

        self.shape = shape ##H, W
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1)/2
        self.conv = nn.Conv2d(self.input_channels, self.num_features, self.filter_size, 1, self.padding)

    def forward(self, state_input_layer1, state_input_layer2):
        """
        Convlutional Decoder for ConvLSTM RNN, forward pass
        """
        inputlayer = torch.cat((state_input_layer1, state_input_layer2),1)
        output = self.conv(inputlayer)

        return output 

class PredModel(nn.Module):
    """
    Overall model with both encoder and decoder part
    """
    def __init__(self, CLSTMargs, decoderargs):
        super(PredModel, self).__init__()

        self.conv_lstm_shape = CLSTMargs[0]
        self.conv_lstm_inp_chans = CLSTMargs[1]
        self.conv_lstm_filter_size = CLSTMargs[2]
        self.conv_lstm_num_features = CLSTMargs[3]
        self.conv_lstm_nlayers = CLSTMargs[4]
        self.conv_lstm = CLSTM( self.conv_lstm_shape, 
                                self.conv_lstm_inp_chans, 
                                self.conv_lstm_filter_size, 
                                self.conv_lstm_num_features, 
                                self.conv_lstm_nlayers)
        self.conv_lstm.apply(weights_init)
        self.conv_lstm.cuda()

        self.decoder_shape = decoderargs[0]
        self.decoder_num_features = decoderargs[1]
        self.decoder_filter_size = decoderargs[2]
        self.decoder_stride = decoderargs[3]
        self.decoder = MNISTDecoder(self.decoder_shape, 
                                    self.decoder_num_features, 
                                    self.decoder_filter_size, 
                                    self.decoder_stride)
        self.decoder.apply(weights_init)
        self.decoder.cuda()

    def forward(self, input, hidden_state):
        out = self.conv_lstm(input, hidden_state)
        pred = self.decoder(out[0][0][0], out[0][0][1])
        return pred

    def init_hidden(self, batch_size):
        init_states = self.conv_lstm.init_hidden(batch_size)
        return init_states

def crossentropyloss(pred, target):
    loss = -torch.sum(torch.log(pred)*target + torch.log(1-pred)*(1-target))
    return loss

###########Usage#######################################    
mnistdata = MovingMNISTdataset("mnist_test_seq.npy")
batch_size = 10

CLSTM_num_features=64
CLSTM_filter_size=5
CLSTM_shape=(64,64)#H,W
CLSTM_inp_chans=1
CLSTM_nlayers=2

CLSTMargs = [CLSTM_shape, CLSTM_inp_chans, CLSTM_filter_size, CLSTM_num_features, CLSTM_nlayers]

decoder_shape = (64, 64)
decoder_num_features = CLSTM_nlayers*CLSTM_num_features
decoder_filter_size = 5
decoder_stride = 1

decoderargs = [decoder_shape, decoder_num_features, decoder_filter_size, decoder_stride]


def main():
    '''
    main function to run the training
    '''
    net = PredModel(CLSTMargs, decoderargs)
    #lossfunction = nn.MSELoss().cuda()
    optimizer = optim.RMSprop(net.parameters(), lr = 0.001)

    hidden_state = net.init_hidden(batch_size)

    for echo in xrange(1):

        for n in xrange(700):

            getitem = mnistdata.__getitem__(n, mode = "train")#shape of 20 10 1 64 64, seq, batch, inpchan, shape
            total = 0

            for i in xrange(10):
                input = getitem[i:i+9, ...].cuda()
                label = getitem[i+10, ...].cuda()

                optimizer.zero_grad()

                hidden_state = net.init_hidden(batch_size)
                pred = net(input, hidden_state)

                pred = F.sigmoid(pred.view(10, -1))
                label = F.sigmoid(pred.view(10, -1))

                #loss = lossfunction(pred, label)
                loss = crossentropyloss(pred, label)
                total += loss
                loss.backward()

                print total

                optimizer.step()

            print "loss: ", total/10
            
if __name__ == "__main__":
    main()