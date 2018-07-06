from Pytorch_RNN import *

mnistdata = MovingMNISTdataset("mnist_test_seq.npy")
batch_size = 10

CLSTM_num_features=32
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