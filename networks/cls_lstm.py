import tensorflow as tf
import numpy as np

import util.helper
import util.global_config as global_config


def generate_cls_gt(input_pipeline_out, frcnn_out):
    target_bbs = input_pipeline_out['target_bbs']

    batch_size = target_bbs.shape[0]
    sequence_size = target_bbs.shape[1]
    PLAYERS = 10

    # The groundtruth bbs with shape (batch_size, backprop_step_size, PLAYERS=10, 4).
    target_bbs = np.reshape(target_bbs, (batch_size, sequence_size, PLAYERS, 4))

    # The RPs with shape (batch_size, sequence_length, PLAYERS=10, 4).
    rp_bbs = frcnn_out

    # The groundtruth for the PLAYERS=10 classifier with shape (batch_size, sequence_size, PLAYERS).
    # Each RP bb is one of the following classes: t from [0, 1, ..., PLAYERS=10].
    # 0 means take LSTM prediction.
    target_cls = np.ones((batch_size, sequence_size, PLAYERS)) * -1

    threshold = 0.6

    for b in range(batch_size):
        for s in range(sequence_size):
            # For the current image.

            # Shape (10, 4).
            current_gt_bbs = target_bbs[b, s]

            # Shape (10, 4).
            current_rp_bbs = rp_bbs[b, s]

            # Go through all players.
            for i in range(PLAYERS):
                # GT for player i. Shape(4).
                gtbb = current_gt_bbs[i]

                ious = np.zeros(PLAYERS)
                # Is there a good region proposal?
                for j in range(PLAYERS):
                    rpbb = current_rp_bbs[j]

                    ious[j] = util.helper.iou(gtbb, rpbb)

                # For player i, the rp bb with highes iou with gt is:
                index = np.argmax(ious)
                if ious[index] >= threshold:
                    target_cls[b, s, i] = index + 1
                else:
                    target_cls[b, s, i] = 0

    return target_cls


# TODO: Remove comments.
def arrange_classifier_inputs(region_proposals, ordered_last_region_proposals, reg_predictions):
    # 1. last_lstm_predictions: [batch_size, sequence_length, 10, 4].
    #       - Add zero vector to have [batch_size, sequence_length + 1, 10, 4]
    #       - Optionally discard last vector.
    # 2. region_proposals (placeholder): [batch_size = 1, sequence_length, 10, 4].
    # 3. last_region_proposals: [batch_size = 1, sequence_length, 10, 4]

    # lstm_predictions Shape (batch_size, step_size, output_dimension).

    region_proposals = tf.reshape(region_proposals, [global_config.cfg['batch_size'],
                                                     global_config.cfg['backprop_step_size'],
                                                     40])

    ordered_last_region_proposals = tf.reshape(ordered_last_region_proposals, [global_config.cfg['batch_size'],
                                                                               global_config.cfg['backprop_step_size'],
                                                                               40])

    ordered_lstm_predictions = tf.reshape(reg_predictions, [global_config.cfg['batch_size'],
                                                            global_config.cfg['backprop_step_size'],
                                                            40])

    # start_region_proposal = tf.zeros((global_config.cfg['batch_size'], 1, 10, 4))
    # last_region_proposals = tf.concat([start_region_proposal, region_proposals[:, :-1]], axis=1)
    # last_region_proposals = tf.reshape(last_region_proposals, [global_config.cfg['batch_size'],
    #                                                           global_config.cfg['backprop_step_size'],
    #                                                           40])

    # start_lstm_prediction = tf.zeros((global_config.cfg['batch_size'], 1, 40))
    # last_lstm_predictions = tf.concat([start_lstm_prediction, lstm_predictions[:, :-1]], axis=1)

    # Stack all into input vector for classificator.
    # cls_input = tf.concat([current_region_proposals, last_region_proposals, last_lstm_predictions], axis=2)

    cls_input = tf.concat([region_proposals, ordered_last_region_proposals, ordered_lstm_predictions], axis=2)

    return cls_input


def _shapeinfo(cls_input, cls_target):
    input_shape = cls_input.get_shape()

    batch_size = input_shape[0]
    step_size = input_shape[1]
    input_dimension = input_shape[2]

    return batch_size, step_size, input_dimension


def network(cls_input, cls_target, state_size, num_layers=1):
    """
    :param cls_input: Shape (batch_size, sequence_size, 40 * 3).
    :param cls_target: Shape (batch_size, sequence_size, 10). NOTE: This means 10 classificators.
    """
    batch_size, step_size, input_dimension = _shapeinfo(cls_input, cls_target)

    multirnn_cell_fw = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.LSTMCell(state_size, use_peepholes=True) for _ in range(num_layers)])
    multirnn_cell_bw = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.LSTMCell(state_size, use_peepholes=True) for _ in range(num_layers)])

    # rnn_outputs = (output_fw, output_bw), each with shape (batch_size, step_size, state_size).
    # final_states = (output_state_fw, output_state_bw), each a tuple with c_state and h_state.
    rnn_outputs, final_states = tf.nn.bidirectional_dynamic_rnn(multirnn_cell_fw, multirnn_cell_bw, cls_input,
                                                                dtype=tf.float32)

    # Shape (batch_size, step_size, 2 * state_size).
    rnn_outputs = tf.concat(rnn_outputs, axis=2)

    # Create classifier for each player.
    PLAYERS = 10
    output_dimension = PLAYERS + 1

    all_W = []
    all_b = []

    for i in range(PLAYERS):
        with tf.variable_scope('classificator_player_' + str(i)):
            W = tf.get_variable('W', [2 * state_size, output_dimension])
            b = tf.get_variable('b', [output_dimension], initializer=tf.constant_initializer(0.0))

            all_W.append(W)
            all_b.append(b)

    all_logits = []

    with tf.name_scope('classificator'):
        for i in range(PLAYERS):
            all = tf.reshape(rnn_outputs, [-1, 2 * state_size])
            out = tf.matmul(all, all_W[i]) + all_b[i]

            predictions = tf.reshape(out, [batch_size.value, step_size.value, output_dimension])

            all_logits.append(predictions)

    all_players_loss = []

    for i in range(PLAYERS):
        with tf.name_scope('loss_player_' + str(i)):
            # Shape (batch_size, sequence_size, 10 + 1).
            predictions = all_logits[i]

            # Shape (batch_size, sequence_size). Values from 0 to 10.
            labels = cls_target[:, :, i]

            player_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions, labels=labels))

            all_players_loss.append(player_loss)

    total_loss = tf.add_n(all_players_loss)

    # Make predictions.
    all_predictions = []
    for i in range(PLAYERS):
        all_predictions.append(tf.nn.softmax(all_logits[i]))

    return total_loss, all_predictions


def build(cls_region_proposals,
          cls_ordered_lrp,
          reg_predictions,
          cls_targets):
    cls_input = arrange_classifier_inputs(cls_region_proposals, cls_ordered_lrp, reg_predictions)

    total_loss, predictions = network(cls_input, cls_targets,
                                      global_config.cfg['lstm_cls_state_size'],
                                      global_config.cfg['lstm_cls_layers'])

    return cls_input, total_loss, predictions
