import tensorflow as tf
import numpy as np


class BasicConvLSTMCell(tf.contrib.rnn.RNNCell):

    def __init__(self, shape, num_filters, kernel_size, forget_bias=1.0, input_size = None, state_is_tuple = True, activation = tf.nn.relu, reuse = None):
        self._shape = shape
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        self._size = tf.TensorShape(shape+[self._num_filters])
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return (tf.contrib.rnn.LSTMStateTuple(self._size, self._size)
                if self._state_is_tuple else 2*self._num_units)

    @property
    def output_size(self):
        return self._size

    def __call__(self, inputs, state, scope = None):
        # input = [time, batch_size, row, col, channel]
        with tf.variable_scope(scope or 'basic_convlstm_cell', reuse = self._reuse):
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = array_ops.split(value = state, num_or_size_splits=2, axis = 3)

            inp_channel = inputs.get_shape().as_list()[-1]+self._num_filters
            out_channel = self._num_filters * 4
            concat = tf.concat([inputs, h], axis = 3)

            kernel = tf.get_variable('kernel', shape = self._kernel_size+[inp_channel, out_channel])
            concat = tf.nn.conv2d(concat, filter=kernel, strides=(1, 1, 1, 1), padding = "SAME")

            i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=3)

            new_c = (c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * self._activation(j))
            new_h = self._activation(new_c) * tf.sigmoid(o)
            if self._state_is_tuple:
                new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat([new_c, new_h], 3)
            return new_h, new_state


class ConvLSTMRNN:

    def __init__(self, train_seq, label_frame):
        self.train_seq = train_seq
        self.label_frame = label_frame
        self.time_steps = 19
        self._state_prev = None
        self._prediction = None
        self._optimize = None
        self._error = None

    @property
    def prediction(self):
        if not self._prediction:

            convlstm_stack_1 = BasicConvLSTMCell([64, 64], 1, [3, 3])

            _, self._state_prev = tf.nn.dynamic_rnn(convlstm_stack_1, self.train_seq, dtype = tf.float32, time_major = True)
            _hidden_state, _ = self._state_prev

            conv_layer_1 = tf.layers.conv2d(_hidden_state, filters=1, kernel_size = 1, strides=[1, 1], padding="SAME", activation = tf.nn.relu, kernel_initializer=tf.initializers.ones, bias_initializer=tf.zeros_initializer())
            self._prediction = conv_layer_1

            return self._prediction

    @property
    def optimize(self):
        if not self._optimize:
            loss_mse = tf.losses.mean_squared_error(
                self.label_frame,
                self.prediction,
                weights = 1.0
            )
            optimizer = tf.train.RMSPropOptimizer(0.01)
            self._optimize = optimizer.minimize(loss_mse)
        return self._optimize

    @property
    def error(self):
        if not self._error:
            error_abs = tf.metrics.mean_absolute_error(
                self.label_frame,
                self._prediction,
            )
            return error_abs


def load_data(path):
    data = np.load(path)
    train = data[:, 0:7, :, :]
    test = data[:, 7000:10000, :, :]
    return train, test

def main():

    height = 64
    width = 64
    step_size = 19

    tf.reset_default_graph()

    train_data, test_data = load_data("mnist_test_seq.npy")
    train_data_input = np.reshape(train_data[0:19, :, :, :], [19, 7, 64, 64, 1])
    train_data_output = np.reshape(train_data[19, :, :, :], [7, 64, 64, 1])

    input_seq = tf.placeholder(tf.float32, [19, 7, height, width, 1])
    output_seq = tf.placeholder(tf.float32, [7, height, width, 1])

    model = ConvLSTMRNN(input_seq, output_seq)

    loss = 0

    config = tf.ConfigProto(
        device_count = {"GPU": 0}
    )
    sess = tf.Session(config = config)
    sess.run(tf.global_variables_initializer())
    sess.run(model.optimize, feed_dict={input_seq: train_data_input,
                                        output_seq: train_data_output})

if __name__ == "__main__":
    main()

