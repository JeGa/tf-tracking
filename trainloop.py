import tensorflow as tf
import numpy as np
import logging

import util.helper


def read_input(sess, tensors, input_handles):
    with util.helper.timeit() as input_time:
        # Images are normalized.
        out_input_pipeline = sess.run(tensors,
                                      feed_dict={input_handles['handle']: input_handles['training_handle']})

    return out_input_pipeline, input_time


def predict_frcnn(sequence_images, frcnn):
    sequence_size = sequence_images.shape[0]

    with util.helper.timeit() as frcnn_time:
        out_frcnn = np.zeros((sequence_size, 10, 4))

        # Go through sequence.
        for i in range(sequence_size):
            # [bbs: (10,4), scores: 10]
            out = frcnn.predict((np.expand_dims(sequence_images[i], 0) * 255).astype(np.uint8))
            out_frcnn[i] = out[0]

    return out_frcnn, frcnn_time


def run(sess, tensors, input_handles, train_writer, epoch, saver, globalstep, frcnn, validate=True):
    sess.run(input_handles['training_initializer'])

    step = 0

    # Go through one epoch.
    while True:
        try:
            out_input_pipeline, input_time = read_input(sess, tensors, input_handles)

            # Take first batch element: [sequence_size, height, width, 3].
            sequence_images = out_input_pipeline['input_data']['images'][0]

            out_frcnn, frcnn_time = predict_frcnn(sequence_images, frcnn)

            util.helper.draw_bb_and_save(sequence_images * 255,
                                         np.reshape(out_input_pipeline['input_data']['groundtruth_bbs'][0],
                                                    (sequence_images.shape[0], 10, 4)),
                                         out_frcnn)

            if step % 1 == 0:
                logging.info('Epoch %d, step %d, global step %d (%.3f/%.3f sec input/frcnn).' % (
                    epoch, step, globalstep, input_time.time(), frcnn_time.time()))

            # if step % global_config.cfg['summary_interval'] == 0:
            #    train_writer.add_summary(out[1], globalstep)

            # if step % global_config.cfg['save_interval'] == 0:
            #    saver.save(sess, os.path.join(global_config.cfg['checkpoints'], 'checkpoint'),
            #               global_step=globalstep)

            # if validate:
            #    if globalstep % global_config.cfg['validation_interval'] == 0:
            #        validation_loop(sess, tensors, input_handles, train_writer, globalstep)

            step += 1
            globalstep += 1
        except tf.errors.OutOfRangeError:
            break

    return globalstep
