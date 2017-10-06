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

    :return total_loss
    :return train_step
    :return predictions: Shape (batch_size, step_size, output_dimension).
    """

    # Deduce shape information from input tensor.
    batch_size, step_size, input_dimension, output_dimension = _shapeinfo(inputs, targets)

    multirnn_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(state_size) for _ in range(num_layers)])

    init_state = multirnn_cell.zero_state(batch_size, dtype=tf.float32)

    # rnn_outputs is of shape (batch_size, step_size, state_size), final_state is a tuple with c_state and h_state.
    rnn_outputs, final_state = tf.nn.dynamic_rnn(multirnn_cell, inputs, initial_state=init_state)

    with tf.variable_scope('linear_regression'):
        W = tf.get_variable('W', [state_size, output_dimension])
        b = tf.get_variable('b', [output_dimension], initializer=tf.constant_initializer(0.0))

    with tf.name_scope('linear_regression_predictor'):
        all = tf.reshape(rnn_outputs, [-1, state_size])
        out = tf.matmul(all, W) + b
        predictions = tf.reshape(out, [batch_size.value, step_size.value, output_dimension.value])

    with tf.name_scope('loss'):
        total_loss = tf.reduce_mean(tf.nn.l2_loss(predictions - targets))

    with tf.name_scope('training'):
        train_step = tf.train.AdagradOptimizer(0.001).minimize(total_loss)

    tf.summary.histogram('state_hist', rnn_outputs)
    tf.summary.histogram('prediction_hist', predictions)

    return total_loss, train_step, predictions
