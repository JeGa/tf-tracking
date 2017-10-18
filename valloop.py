import numpy as np
import tensorflow as tf

import trainloop
import util.global_config as global_config
import util.helper
import util.network_io_utils

# TODO: Hacky, Hacky.
frcnn_out = None
frcnn_time = None


def run(sess, input_pipeline_tensors, input_handles, network_tensors, frcnn):
    sess.run(input_handles['validation_initializer'])

    step = 0

    # Go through one epoch.
    while True:
        try:
            input_pipeline_out, input_time = util.network_io_utils.read_input(sess, input_pipeline_tensors,
                                                                              input_handles, 'validation')

            global frcnn_out
            global frcnn_time

            if frcnn_out is None:
                frcnn_out, frcnn_time = util.network_io_utils.predict_frcnn(input_pipeline_out['images'], frcnn)
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
                    util.network_io_utils.sort_rps(frcnn_out, ordered_last_region_proposals,
                                                   output_tensors['cls_predictions'],
                                                   ordered_lstmreg_input,
                                                   output_tensors['reg_predictions'],
                                                   i)

            for s in range(input_pipeline_out['images'].shape[1]):
                util.helper.draw_allbbs_and_cls_labels_and_save_predict(
                    input_pipeline_out['images'][0, s],
                    np.reshape(output_tensors['reg_predictions'][0, s], (10, 4)),
                    frcnn_out[0, s],
                    util.network_io_utils.cls_pred(output_tensors['cls_predictions'], 0, s),
                    'predict_batch0_time' + str(s))

            step += 1

        except tf.errors.OutOfRangeError:
            break
