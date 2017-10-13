import tensorflow as tf
import numpy as np
import logging
import os

import util.helper
import networks.player_classificator
import util.global_config as global_config


# If you want to run only the input data reading part.
def read_input(sess, tensors, input_handles):
    with util.helper.timeit() as input_time:
        # Images are normalized.
        out_input_pipeline = sess.run(tensors, feed_dict={input_handles['handle']: input_handles['training_handle']})

    return out_input_pipeline, input_time.time()


# Uses an own session for the rcnn graph.
def predict_frcnn(sequence_images, frcnn):
    batch_size = sequence_images.shape[0]
    sequence_length = sequence_images.shape[1]

    frcnn_out = np.zeros((batch_size,
                          sequence_length,
                          10, 4))

    with util.helper.timeit() as frcnn_time:
        for i in range(batch_size):
            for j in range(sequence_length):
                frcnn_out[i][j], _ = frcnn.predict((np.expand_dims(sequence_images[i][j], 0) * 255).astype(np.uint8))

    return frcnn_out, frcnn_time.time()


# Train the lstm with input data given as feed_dict.
def train(sess, tensors, input_pipeline_out, frcnn_out, target_cls):
    input_tensors = {
        'train_step': tensors['combined']['train_step'],
        'total_loss': tensors['combined']['loss'],

        'cls_inputs': tensors['cls']['inputs'],
        'cls_targets': tensors['cls']['targets'],
        # This is a list.
        'cls_predictions': tensors['cls']['predictions'],
        # Shifted lstm pred. from 'lstm' part.
        'last_lstm_predictions': tensors['cls']['last_lstm_predictions'],

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
            tensors['placeholders']['target_cls']: target_cls
        })

    return out, train_time.time()


def interval_actions(epoch, step, globalstep,
                     input_time, frcnn_time, train_time,
                     loss, train_writer, summary, saver, sess):
    # sequence_images, out_input_pipeline, out_frcnn, out_lstm):
    if step % 1 == 0:
        logging.info(
            'Epoch %d, step %d, global step %d (%.3f/%.3f/%.3f sec input/frcnn_predict/train). Loss %.3f.' % (
                epoch, step, globalstep, input_time, frcnn_time, train_time, loss))

    if step % global_config.cfg['summary_interval'] == 0:
        train_writer.add_summary(summary, globalstep)

    if step % global_config.cfg['result_interval'] == 0:
        pass

    if step % global_config.cfg['save_interval'] == 0:
        saver.save(sess, os.path.join(global_config.cfg['checkpoints'], 'checkpoint'),
                   global_step=globalstep)

        # if validate:
        #    if globalstep % global_config.cfg['validation_interval'] == 0:
        #        validation_loop(sess, tensors, input_handles, train_writer, globalstep)


def run(sess, input_pipeline_tensors, input_handles, network_tensors,
        train_writer, epoch, saver, globalstep, frcnn, validate=True):
    """
    input_pipeline_tensors.keys():
        'groundtruth_bbs'
        'target_bbs'
        'images'

    network_tensors.keys():
      'input_data_placeholders': {
          'groundtruth_bbs': groundtruth_bbs,
          'target_bbs': target_bbs,
          'images': images
      }
      'lstm': {
          'inputs': inputs,
          'targets': targets,
          'total_loss': total_loss,
          'train_step': train_step,
          'predictions': predictions
      }
      'summary'

    input_handles.keys():
          'training_initializer'
          'validation_initializer'
          'training_handle'
          'validation_handle'
          'handle'
    """
    sess.run(input_handles['training_initializer'])

    step = 0

    # Go through one epoch.
    while True:
        try:
            input_pipeline_out, input_time = read_input(sess, input_pipeline_tensors, input_handles)

            frcnn_out, frcnn_time = predict_frcnn(input_pipeline_out['images'], frcnn)

            # Shape (batch_size, sequence_size, 10).
            target_cls = networks.player_classificator.generate_cls_gt(input_pipeline_out, frcnn_out)

            # Draw the gt labels.
            # util.helper.draw_bb_and_cls_labels_and_save(input_pipeline_out['images'][0, 0],
            #                                             frcnn_out[0, 0],
            #                                             np.reshape(input_pipeline_out['target_bbs'][0, 0], (10, 4)),
            #                                             target_cls[0, 0],
            #                                             '0_0_gt')

            out, train_time = train(sess, network_tensors, input_pipeline_out, frcnn_out, target_cls)

            interval_actions(epoch, step, globalstep,
                             input_time, frcnn_time, train_time,
                             out['total_loss'], train_writer, out['summary'],
                             saver, sess)

            def cls_pred(predictions, batch, time):
                classifications = np.zeros((10, 11))
                for i in range(len(predictions)):
                    classifications[i] = predictions[i][batch, time]
                return classifications

            # Draw the predicted classification labels.
            if step % 5 == 0:
                for s in range(input_pipeline_out['images'].shape[1]):
                    util.helper.draw_allbbs_and_cls_labels_and_save(
                        input_pipeline_out['images'][0, s],
                        np.reshape(out['reg_targets'][0, s], (10, 4)),
                        np.reshape(out['last_lstm_predictions'][0, s], (10, 4)),
                        frcnn_out[0, s],
                        out['cls_targets'][0, s],
                        cls_pred(out['cls_predictions'], 0, s),
                        'batch0_time' + str(s))

            step += 1
            globalstep += 1
        except tf.errors.OutOfRangeError:
            break

    return globalstep
