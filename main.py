from Pytorch_RNN import *
import argparse

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
    basecell = 'CLSTM'

if args.MSELoss:
    objectfunction = 'MSELoss'
if args.crossentropyloss:
    objectfunction = 'crossentropyloss'
else:
    objectfunction = 'crossentropyloss'



mnistdata = MovingMNISTdataset("mnist_test_seq.npy")
batch_size = 10

CRNN_num_features=32
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


def main():
    '''
    main function to run the training
    '''
    net = PredModel(CRNNargs, decoderargs, cell = basecell)

    if objectfunction == 'MSELoss':
        lossfunction = nn.MSELoss().cuda()
    
    optimizer = optim.RMSprop(net.parameters(), lr = 0.001)

    hidden_state = net.init_hidden(batch_size)

    for epoch in xrange(1):

        for n in xrange(700):

            getitem = mnistdata.__getitem__(n, mode = "train")#shape of 20 10 1 64 64, seq, batch, inpchan, shape
            total = 0

            for i in xrange(10):
                input = getitem[i:i+9, ...].cuda()
                label = getitem[i+10, ...].cuda()

                optimizer.zero_grad()

                hidden_state = net.init_hidden(batch_size)
                pred = net(input, hidden_state)

                if objectfunction == 'MSELoss':
                    loss = lossfunction(pred, label)
                if objectfunction == 'crossentropyloss':
                    pred = F.sigmoid(pred.view(10, -1))
                    label = label.view(10, -1)
                    loss = crossentropyloss(pred, label)
                total += loss
                loss.backward()

                optimizer.step()

            print "loss: ", total.item()/10*1000, "E-3   epoch: ", epoch

if __name__ == "__main__":
    main()