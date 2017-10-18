import tensorflow as tf
import numpy as np

import trainloop
import util.global_config as global_config
import util.helper

# TODO: Hacky, Hacky.
frcnn_out = None
frcnn_time = None


def read_input(sess, tensors, input_handles):
    with util.helper.timeit() as input_time:
        # Images are normalized.
        out_input_pipeline = sess.run(tensors, feed_dict={input_handles['handle']: input_handles['validation_handle']})

    return out_input_pipeline, input_time.time()


def sort_rps(frcnn_out, ordered_last_region_proposals,
             cls_predictions,
             ordered_lstmreg_input,
             reg_predictions,
             t):
    """
    :param: frcnn_out: Shape (batch_size, sequence_length, 10, 4).

    :param: ordered_lstmreg_input: Shape (batch_size, sequence_length, 10, 4).
    :param: ordered_last_region_proposals: Shape (batch_size, sequence_length, 10, 4).
    :param: ordered_current_lstmreg: Shape (batch_size, sequence_length, 10, 4).

    :param: cls_predictions: List of 10 elements with shape (batch_size, sequence_length, 11).
    :param: reg_predictions: Shape (batch_size, sequence_length, 40).
    """
    batch_size = frcnn_out.shape[0]

    reg_predictions = np.reshape(reg_predictions, (batch_size, -1, 10, 4))

    for j in range(batch_size):
        # 0 means take lstm. Shape (10).
        current_cls_prediction = trainloop.cls_pred_final(cls_predictions, j, t)

        if t == 0:
            ordered_last_region_proposals[j, t + 1] = frcnn_out[j, t]
            ordered_lstmreg_input[j, t + 1] = frcnn_out[j, t]
        else:
            for i in range(10):
                # i = player id.
                predid = current_cls_prediction[i]

                if predid == 0:
                    # Take lstm.
                    ordered_lstmreg_input[j, t + 1, i] = reg_predictions[j, t, i]
                else:
                    # Take rp.
                    ordered_lstmreg_input[j, t + 1, i] = frcnn_out[j, t, predid - 1]

            ordered_last_region_proposals[j, t + 1] = ordered_lstmreg_input[j, t + 1]


def run(sess, input_pipeline_tensors, input_handles, network_tensors,
        train_writer, epoch, saver, globalstep, frcnn,
        cls_weight, reg_weight):
    sess.run(input_handles['validation_initializer'])

    step = 0

    # Go through one epoch.
    while True:
        try:
            input_pipeline_out, input_time = read_input(sess, input_pipeline_tensors, input_handles)

            global frcnn_out
            global frcnn_time

            if frcnn_out is None:
                frcnn_out, frcnn_time = trainloop.predict_frcnn(input_pipeline_out['images'], frcnn)
            else:
                frcnn_time = 0

            # ============================================================================================
            # DO THE STUFF HERE.

            batch_size = global_config.cfg['batch_size']
            sequence_length = global_config.cfg['backprop_step_size']

            # The required shape.
            ordered_lstmreg_input = np.zeros((batch_size, sequence_length, 10, 4))

            ordered_last_region_proposals = np.zeros((batch_size, sequence_length, 10, 4))

            # Initialize.
            ordered_lstmreg_input[:, 0] = frcnn_out[:, 0, :, :]

            # np.zeros((batch_size, 10, 4))
            # ordered_lstmreg[:, 0] = np.zeros((batch_size, 10, 4))

            output_tensors = None

            for i in range(sequence_length):
                input_tensors = {
                    'cls_predictions': network_tensors['cls']['predictions'],
                    'reg_predictions': network_tensors['lstm']['predictions']
                }

                # Prediction.
                output_tensors = sess.run(input_tensors, feed_dict={
                    # Reg part.
                    network_tensors['placeholders']['groundtruth_bbs']: np.reshape(
                        ordered_lstmreg_input, (batch_size, sequence_length, 40)),

                    # Cls part.
                    network_tensors['placeholders']['region_proposals']: frcnn_out,
                    network_tensors['placeholders']['ordered_last_region_proposals']: ordered_last_region_proposals,
                })

                if i < sequence_length - 1:
                    # Sort inputs.
                    sort_rps(frcnn_out, ordered_last_region_proposals,
                             output_tensors['cls_predictions'],
                             ordered_lstmreg_input,
                             output_tensors['reg_predictions'],
                             i)

            for s in range(input_pipeline_out['images'].shape[1]):
                util.helper.draw_allbbs_and_cls_labels_and_save_predict(
                    input_pipeline_out['images'][0, s],
                    np.reshape(output_tensors['reg_predictions'][0, s], (10, 4)),
                    frcnn_out[0, s],
                    trainloop.cls_pred(output_tensors['cls_predictions'], 0, s),
                    'predict_batch0_time' + str(s))

            step += 1

        except tf.errors.OutOfRangeError:
            break
