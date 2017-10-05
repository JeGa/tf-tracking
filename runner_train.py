import tensorflow as tf
import click
import logging
import time
import os

import util.global_config as global_config
import util.builder
import util.helper

import networks.faster_rcnn_odapi_loader

from PIL import Image
from PIL import ImageDraw
import os
import numpy as np

logging.basicConfig(level=logging.INFO)


# def validation_loop(sess, tensors, input_handles, train_writer, globalstep):
#     logging.info('Run validation.')
#
#     sess.run(input_handles['validation_initializer'])
#
#     validation_loss = []
#     while True:
#         try:
#             out_val = sess.run([tensors['loss']],
#                                feed_dict={input_handles['handle']: input_handles['validation_handle']})
#             validation_loss.append(out_val[0])
#         except tf.errors.OutOfRangeError:
#             break
#
#     sumloss = sum(validation_loss) / max(len(validation_loss), 1)
#     logging.info('Validation loss: ' + str(sumloss))
#
#     summary = tf.Summary()
#     summary.value.add(tag='total_validation_loss', simple_value=sumloss)
#     train_writer.add_summary(summary, globalstep)


def train_loop(sess, tensors, input_handles, train_writer, epoch, saver, globalstep, frcnn, validate=True):
    sess.run(input_handles['training_initializer'])

    step = 0

    # Go through one epoch.
    while True:
        try:
            with util.helper.timeit() as input_time:
                out_input_pipeline = sess.run(tensors,
                                              feed_dict={input_handles['handle']: input_handles['training_handle']})

            with util.helper.timeit() as frcnn_time:
                # [bbs: (10,4), scores: 10]
                out_frcnn = frcnn.predict((out_input_pipeline['input_data']['image'] * 255).astype(np.uint8))

            util.helper.draw_bb_and_save(out_input_pipeline['input_data']['image'] * 255, out_frcnn[0])

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


def loop(sess, tensors, input_handles, frcnn):
    filename = util.helper.summary_file([global_config.cfg['batch_size'],
                                         global_config.cfg['epochs']])
    train_writer = tf.summary.FileWriter(filename, sess.graph)

    # TODO
    saver = None  # tf.train.Saver()

    globalstep = 0
    with util.helper.timeit() as ttime:
        for epoch in range(global_config.cfg['epochs']):
            globalstep = train_loop(sess, tensors, input_handles, train_writer, epoch, saver, globalstep, frcnn)
    logging.info('Done training (' + str(ttime.time()) + ' sec, ' + str(globalstep) + ' steps).')


@click.command()
@click.option("--config", default="config.yml", help="The configuration file.")
def main(config):
    # This makes the configuration available as global_config.cfg dictionary.
    global_config.read(config)

    frcnn = networks.faster_rcnn_odapi_loader.faster_rcnn_odapi()
    frcnn.import_graph(global_config.cfg['faster_rcnn_graph'])

    with tf.Graph().as_default():
        # Build the computational graph.
        tensors, input_handles = util.builder.build()

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        with tf.Session() as sess:
            sess.run(init_op)
            training_handle = sess.run(input_handles['tr_h'])
            validation_handle = sess.run(input_handles['val_h'])

            # This is required for switching between training and validation phase.
            input_handles_eval = {
                'training_initializer': input_handles['tr_it'].initializer,
                'validation_initializer': input_handles['val_it'].initializer,
                'training_handle': training_handle,
                'validation_handle': validation_handle,
                'handle': input_handles['h']
            }

            loop(sess, tensors, input_handles_eval, frcnn)


if __name__ == '__main__':
    main()
