from Pytorch_RNN import *
from torch.utils.data import DataLoader
import argparse
import os

parser = argparse.ArgumentParser(description = "Determine the Type of Cells and Loss Function to be Used")
parser.add_argument('-clstm','--convlstm', help = 'use convlstm as base cell', action = 'store_true')
parser.add_argument('-cgru','--convgru', help = 'use convgru as base cell', action = 'store_true')
parser.add_argument('-MSE', '--MSELoss', help = 'use MSE as loss function', action = 'store_true')
parser.add_argument('-xentro', '--crossentropyloss', help = 'use Cross Entropy Loss as loss function', action = 'store_true')
args = parser.parse_args()

if args.convlstm:
    basecell = 'CLSTM'
if args.convgru:
    basecell = 'CGRU'
else:
    basecell = 'CGRU'

if args.MSELoss:
    objectfunction = 'MSELoss'
if args.crossentropyloss:
    objectfunction = 'crossentropyloss'
else:
    objectfunction = 'crossentropyloss'


###Dataset and Dataloader
batch_size = 20
mnistdata = MovingMNISTdataset("mnist_test_seq.npy")
trainingdata_loader = DataLoader(dataset = mnistdata, batch_size = batch_size, shuffle=True)

CRNN_num_features=128
CRNN_filter_size=5
CRNN_shape=(64,64)#H,W
CRNN_inp_chans=1
CRNN_nlayers=2

CRNNargs = [CRNN_shape, CRNN_inp_chans, CRNN_filter_size, CRNN_num_features, CRNN_nlayers]

decoder_shape = (64, 64)
decoder_num_features = CRNN_nlayers*CRNN_num_features
decoder_filter_size = 5
decoder_stride = 1

decoderargs = [decoder_shape, decoder_num_features, decoder_filter_size, decoder_stride]

def train():
    '''
    main function to run the training
    '''
    net = PredModel(CRNNargs, decoderargs, cell = basecell)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count()>1:
        net = nn.DataParallel(net)
        multipleDevice = True
    else:
        multipleDevice = False

    net.to(device)

    if objectfunction == 'MSELoss':
        lossfunction = nn.MSELoss().cuda()
    
    optimizer = optim.RMSprop(net.parameters(), lr = 0.0002)

    if multipleDevice:
        hidden_state = net.module.init_hidden(batch_size)
    else:
        hidden_state = net.init_hidden(batch_size)

    for epoch in xrange(1):

        for data in trainingdata_loader:

            total = 0

            input = data[:, 0:10, ...].to(device)
            label = data[:, 10:20, ...].to(device)

            optimizer.zero_grad()

            if multipleDevice:
                hidden_state = net.module.init_hidden(batch_size)
            else:
                hidden_state = net.init_hidden(batch_size)

            pred = net(input, hidden_state)

            if objectfunction == 'MSELoss':
                loss = 0
                for seq in range(len(label)):
                    loss += lossfunction(pred[seq], label[seq])

            if objectfunction == 'crossentropyloss':
                loss = 0
                for seq in range(10):
                    predframe = F.sigmoid(pred[seq].view(batch_size, -1))
                    labelframe = label[:, seq, ...].view(batch_size, -1)
                    loss += crossentropyloss(predframe, labelframe)
                
            print "loss: ", loss, "  epoch :", epoch
            loss.backward()
            optimizer.step()

    save_path = os.getcwd()
    torch.save(net, save_path+"/trained_model")

def test():
    file_path = os.getcwd() + '/trained_model'
    testnet = torch.load(file_path)

def inference():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count()>1:
        net = nn.DataParallel(net)
        multipleDevice = True
    else:
        multipleDevice = False

    file_path = os.getcwd() + '/trained_model'
    inferencenet = torch.load(file_path)

    for data in trainingdata_loader:
        input = data[:, 0:10, ...].to(device)
        if multipleDevice:
            hidden_state = inferencenet.module.init_hidden(batch_size)
        else:
            hidden_state = inferencenet.init_hidden(batch_size)

        pred = inferencenet(input, hidden_state)
        break

    np.save(os.getcwd()+'/input', input)
    np.save(os.getcwd()+'/label', data[:, 10:20, ...])
    np.save(os.getcwd()+'/inference', pred)

if __name__ == "__main__":
    train()
    inference()
