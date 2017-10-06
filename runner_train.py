import tensorflow as tf
import click
import logging

import util.global_config as global_config
import util.builder
import util.helper

import trainloop

import networks.faster_rcnn_odapi_loader

logging.basicConfig(level=logging.INFO)


def loop(sess, tensors, input_handles, frcnn):
    filename = util.helper.summary_file([global_config.cfg['batch_size'],
                                         global_config.cfg['epochs']])
    train_writer = tf.summary.FileWriter(filename, sess.graph)

    # TODO
    saver = None  # tf.train.Saver()

    globalstep = 0
    with util.helper.timeit() as ttime:
        for epoch in range(global_config.cfg['epochs']):
            globalstep = trainloop.run(sess, tensors, input_handles, train_writer, epoch, saver, globalstep, frcnn)
    logging.info('Done training (' + str(ttime.time()) + ' sec, ' + str(globalstep) + ' steps).')


@click.command()
@click.option("--config", default="config.yml", help="The configuration file.")
def main(config):
    # This makes the configuration available as global_config.cfg dictionary and makes the required folders.
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
