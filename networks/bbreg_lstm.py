import tensorflow as tf


def _shapeinfo(inputs, targets):
    input_shape = inputs.get_shape()

    batch_size = input_shape[0]
    step_size = input_shape[1]
    input_dimension = input_shape[2]

    output_dimension = targets.get_shape()[2]

    return batch_size, step_size, input_dimension, output_dimension


def build(inputs, targets, state_size, num_layers=1):
    """
    :param inputs: Shape (batch_size, step_size, input_dimension).
    :param targets: Shape (batch_size, step_size, input_dimension).
    :param state_size: Scalar.
    :param num_layers: Scalar.
    :param learning_rate: Scalar

    :return total_loss
    :return train_step
    :return predictions: Shape (batch_size, step_size, output_dimension).
    """
    # Deduce shape information from input tensor.
    batch_size, step_size, input_dimension, output_dimension = _shapeinfo(inputs, targets)

    multirnn_cell_fw = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.LSTMCell(state_size, use_peepholes=True) for _ in range(num_layers)])
    multirnn_cell_bw = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.LSTMCell(state_size, use_peepholes=True) for _ in range(num_layers)])

    # rnn_outputs = (output_fw, output_bw), each with shape (batch_size, step_size, state_size).
    # final_states = (output_state_fw, output_state_bw), each a tuple with c_state and h_state.
    rnn_outputs, final_states = tf.nn.bidirectional_dynamic_rnn(multirnn_cell_fw, multirnn_cell_bw, inputs,
                                                                dtype=tf.float32)

    # Shape (batch_size, step_size, 2 * state_size).
    rnn_outputs = tf.concat(rnn_outputs, axis=2)

    with tf.variable_scope('linear_regression'):
        W = tf.get_variable('W', [2 * state_size, output_dimension])
        b = tf.get_variable('b', [output_dimension], initializer=tf.constant_initializer(0.0))

    with tf.name_scope('predictor'):
        all = tf.reshape(rnn_outputs, [-1, 2 * state_size])
        out = tf.matmul(all, W) + b
        predictions = tf.reshape(out, [batch_size.value, step_size.value, output_dimension.value])

    with tf.name_scope('loss'):
        total_loss = tf.reduce_mean(tf.nn.l2_loss(predictions - targets))

    # TODO
    # with tf.name_scope('training'):
    #    train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

    tf.summary.histogram('state_hist', rnn_outputs)
    tf.summary.histogram('prediction_hist', predictions)

    return total_loss, predictions
