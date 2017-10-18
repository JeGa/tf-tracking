import logging
import os
import tensorflow as tf
import numpy as np

import networks.cls_lstm
import util.global_config as global_config
import util.helper
import util.network_io_utils
import valloop

# TODO: Hacky, Hacky.
frcnn_out = None
frcnn_time = None


# Train the lstm with input data given as feed_dict.
def train(sess, tensors, input_pipeline_out, frcnn_out, target_cls, ordered_last_region_proposals,
          cls_weight=0.5, reg_weight=0.5):
    input_tensors = {
        'train_step': tensors['combined']['train_step'],
        'total_loss': tensors['combined']['loss'],

        'cls_inputs': tensors['cls']['inputs'],
        'cls_targets': tensors['cls']['targets'],
        # This is a list.
        'cls_predictions': tensors['cls']['predictions'],

        'reg_targets': tensors['lstm']['targets'],
        'reg_predictions': tensors['lstm']['predictions'],

        'summary': tensors['summary']
    }

    with util.helper.timeit() as train_time:
        out = sess.run(input_tensors, feed_dict={
            tensors['placeholders']['groundtruth_bbs']: input_pipeline_out['groundtruth_bbs'],
            tensors['placeholders']['target_bbs']: input_pipeline_out['target_bbs'],
            tensors['placeholders']['images']: input_pipeline_out['images'],

            tensors['placeholders']['region_proposals']: frcnn_out,
            tensors['placeholders']['target_cls']: target_cls,
            tensors['placeholders']['ordered_last_region_proposals']: ordered_last_region_proposals,

            tensors['combined']['cls_weight']: cls_weight,
            tensors['combined']['reg_weight']: reg_weight
        })

    return out, train_time.time()


def interval_actions(epoch, step, globalstep,
                     input_time, frcnn_time, train_time,
                     loss, train_writer, summary, saver, sess):
    # sequence_images, out_input_pipeline, out_frcnn, out_lstm):
    if globalstep % 1 == 0:
        logging.info(
            'Epoch %d, step %d, global step %d (%.3f/%.3f/%.3f sec input/frcnn_predict/train). Loss %.3f.' % (
                epoch, step, globalstep, input_time, frcnn_time, train_time, loss))

    if globalstep % global_config.cfg['summary_interval'] == 0:
        train_writer.add_summary(summary, globalstep)

    if globalstep % global_config.cfg['result_interval'] == 0:
        pass

    if globalstep % global_config.cfg['save_interval'] == 0:
        saver.save(sess, os.path.join(global_config.cfg['checkpoints'], 'checkpoint'),
                   global_step=globalstep)


def run(sess, input_pipeline_tensors, input_handles, network_tensors,
        train_writer, epoch, saver, globalstep, frcnn,
        cls_weight, reg_weight, validate=True):
    sess.run(input_handles['training_initializer'])

    step = 0

    # Go through one epoch.
    while True:
        try:
            input_pipeline_out, input_time = util.network_io_utils.read_input(sess, input_pipeline_tensors,
                                                                              input_handles, 'training')

            global frcnn_out
            global frcnn_time

            if frcnn_out is None:
                frcnn_out, frcnn_time = util.network_io_utils.predict_frcnn(input_pipeline_out['images'], frcnn)
            else:
                frcnn_time = 0

            # ============================================================================================
            # DO THE STUFF HERE.

            # Shape (batch_size, sequence_size, 10).
            target_cls = networks.cls_lstm.generate_cls_gt(input_pipeline_out, frcnn_out)

            # TODO: Does this even make sense???

            # Generate ordered_last_region_proposals with shape (batch_size, sequence_length, 10, 4).
            batch_size = global_config.cfg['batch_size']
            sequence_length = global_config.cfg['backprop_step_size']

            ordered_last_region_proposals = np.zeros((batch_size, sequence_length, 10, 4))
            for t in range(sequence_length - 1):
                for j in range(batch_size):
                    gtbb = np.reshape(input_pipeline_out['groundtruth_bbs'][j, t], (10, 4))

                    if t == 0:
                        ordered_last_region_proposals[j, t + 1] = frcnn_out[j, t]
                    else:
                        for i in range(10):
                            # i = player id.
                            predid = int(target_cls[j, t, i])

                            if predid == 0:
                                # Take lstm.
                                ordered_last_region_proposals[j, t + 1, i] = gtbb[i]
                            else:
                                # Take rp.
                                ordered_last_region_proposals[j, t + 1, i] = frcnn_out[j, t, predid - 1]

            out, train_time = train(sess, network_tensors, input_pipeline_out, frcnn_out, target_cls,
                                    ordered_last_region_proposals,
                                    cls_weight, reg_weight)

            interval_actions(epoch, step, globalstep,
                             input_time, frcnn_time, train_time,
                             out['total_loss'], train_writer, out['summary'],
                             saver, sess)

            if validate:
                if globalstep % global_config.cfg['validation_interval'] == 0:
                    logging.info('**Start validation.**')
                    valloop.run(sess, input_pipeline_tensors, input_handles, network_tensors, frcnn)
                    logging.info('**Finished validation.**')

            step += 1
            globalstep += 1
        except tf.errors.OutOfRangeError:
            break

    return globalstep
