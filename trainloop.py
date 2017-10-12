import tensorflow as tf
import numpy as np
import logging

import util.helper
import networks.player_classificator


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


# TODO
# Train the lstm with input data given as feed_dict.
def train(sess, tensors, input_pipeline_out, frcnn_out):
    with util.helper.timeit() as train_time:
        train_lstm_out, summary_out = sess.run([tensors['lstm'], tensors['summary']], feed_dict={
            tensors['placeholders']['groundtruth_bbs']: input_pipeline_out['groundtruth_bbs'],
            tensors['placeholders']['target_bbs']: input_pipeline_out['target_bbs'],
            tensors['placeholders']['images']: input_pipeline_out['images'],

            tensors['placeholders']['region_proposals']: frcnn_out
        })
    return train_lstm_out, summary_out, train_time.time()


# TODO
def interval_actions(epoch, step, globalstep,
                     input_time, frcnn_time, train_time,
                     loss, train_writer, summary):
    # sequence_images, out_input_pipeline, out_frcnn, out_lstm):
    if step % 1 == 0:
        logging.info(
            'Epoch %d, step %d, global step %d (%.3f/%.3f/%.3f sec input/frcnn_predict/train). Loss %.3f.' % (
                epoch, step, globalstep, input_time, frcnn_time, train_time, loss))

        # if step % global_config.cfg['summary_interval'] == 0:
        #   train_writer.add_summary(summary, globalstep)

        # if step % global_config.cfg['result_interval'] == 0:
        #     util.helper.draw_bb_and_save(sequence_images * 255,
        #                                  np.reshape(out_input_pipeline['groundtruth_bbs'][0],
        #                                             (sequence_images.shape[0], 10, 4)),
        #                                  out_frcnn,
        #                                  np.reshape(out_lstm['predictions'][0], (-1, 10, 4)))

        # if step % global_config.cfg['save_interval'] == 0:
        #    saver.save(sess, os.path.join(global_config.cfg['checkpoints'], 'checkpoint'),
        #               global_step=globalstep)

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

            # util.helper.draw_bb_and_cls_labels_and_save(input_pipeline_out['images'][0, 0],
            #                                             frcnn_out[0, 0],
            #                                             np.reshape(input_pipeline_out['target_bbs'][0, 0], (10, 4)),
            #                                             target_cls[0, 0],
            #                                             '0_0')

            lstm_out, summary_out, train_time = train(sess, network_tensors, input_pipeline_out, frcnn_out)

            interval_actions(epoch, step, globalstep,
                             input_time, frcnn_time, train_time,
                             lstm_out['loss'], train_writer, summary_out)

            step += 1
            globalstep += 1
        except tf.errors.OutOfRangeError:
            break

    return globalstep
