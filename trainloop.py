import tensorflow as tf
import numpy as np
import logging

import util.helper
import util.global_config as global_config


# TODO: Unused.
# If you want to run only the input data reading part.
def read_input(sess, input_tensors, input_handles):
    with util.helper.timeit() as input_time:
        # Images are normalized.
        out_input_pipeline = sess.run(input_tensors,
                                      feed_dict={input_handles['handle']: input_handles['training_handle']})

    return out_input_pipeline, input_time


# Uses an own session for the rrcnn graph.
def predict_frcnn(sequence_images, frcnn):
    sequence_size = sequence_images.shape[0]

    with util.helper.timeit() as frcnn_time:
        out_frcnn = np.zeros((sequence_size, 10, 4))

        # Go through sequence.
        for i in range(sequence_size):
            # [bbs: (10,4), scores: 10]
            out = frcnn.predict((np.expand_dims(sequence_images[i], 0) * 255).astype(np.uint8))
            out_frcnn[i] = out[0]

    return out_frcnn, frcnn_time.time()


# Run the complete graph.
def runall(sess, tensors, input_handles):
    with util.helper.timeit() as t:
        out = sess.run(tensors, feed_dict={input_handles['handle']: input_handles['training_handle']})

    return out['input_data'], out['lstm'], out['summary'], t.time()


def interval_actions(epoch, step, globalstep, train_time, frcnn_time, train_writer, summary,
                     sequence_images, out_input_pipeline, out_frcnn, out_lstm):
    if step % 1 == 0:
        logging.info('Epoch %d, step %d, global step %d (%.3f/%.3f sec lstm_train/frcnn_predict).' % (
            epoch, step, globalstep, train_time, frcnn_time))

    if step % global_config.cfg['summary_interval'] == 0:
        train_writer.add_summary(summary, globalstep)

    if step % global_config.cfg['result_interval'] == 0:
        util.helper.draw_bb_and_save(sequence_images * 255,
                                     np.reshape(out_input_pipeline['groundtruth_bbs'][0],
                                                (sequence_images.shape[0], 10, 4)),
                                     out_frcnn,
                                     np.reshape(out_lstm['predictions'][0], (-1, 10, 4)))

        # if step % global_config.cfg['save_interval'] == 0:
        #    saver.save(sess, os.path.join(global_config.cfg['checkpoints'], 'checkpoint'),
        #               global_step=globalstep)

        # if validate:
        #    if globalstep % global_config.cfg['validation_interval'] == 0:
        #        validation_loop(sess, tensors, input_handles, train_writer, globalstep)


def run(sess, tensors, input_handles, train_writer, epoch, saver, globalstep, frcnn, validate=True):
    sess.run(input_handles['training_initializer'])

    step = 0

    # Go through one epoch.
    while True:
        try:
            out_input_pipeline, out_lstm, summary, train_time = runall(sess, tensors, input_handles)

            # Take first batch element: [sequence_size, height, width, 3].
            sequence_images = out_input_pipeline['images'][0]

            # Make frcnn predictions for all images in that sequence.
            out_frcnn, frcnn_time = predict_frcnn(sequence_images, frcnn)

            # Actions that are done only each n steps.
            interval_actions(epoch, step, globalstep, train_time, frcnn_time, train_writer, summary, sequence_images,
                             out_input_pipeline, out_frcnn, out_lstm)

            step += 1
            globalstep += 1
        except tf.errors.OutOfRangeError:
            break

    return globalstep
