import tensorflow as tf
import numpy as np
import logging

import util.helper
import util.global_config as global_config


# If you want to run only the input data reading part.
def read_input(sess, tensors, input_handles):
    with util.helper.timeit() as input_time:
        # Images are normalized.
        out_input_pipeline = sess.run(tensors['input_data'],
                                      feed_dict={input_handles['handle']: input_handles['training_handle']})

    return out_input_pipeline, input_time.time()


# Train the lstm with input data given as feed_dict.
def train(sess, tensors, input_pipeline_out, frcnn_out):
    with util.helper.timeit() as train_time:
        train_lstm_out, summary_out = sess.run([tensors['lstm'], tensors['summary']], feed_dict={
            tensors['input_data_placeholders']['groundtruth_bbs']: input_pipeline_out['groundtruth_bbs'],
            tensors['input_data_placeholders']['target_bbs']: input_pipeline_out['target_bbs'],
            tensors['input_data_placeholders']['images']: input_pipeline_out['images'],

            tensors['rpn']['region_proposals_placeholder']: frcnn_out
        })
    return train_lstm_out, summary_out, train_time.time()


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


def interval_actions(epoch, step, globalstep,
                     input_time, train_time, frcnn_time,
                     loss, train_writer, summary):
    # sequence_images, out_input_pipeline, out_frcnn, out_lstm):
    if step % 1 == 0:
        logging.info(
            'Epoch %d, step %d, global step %d (%.3f/%.3f/%.3f sec input/lstm_train/frcnn_predict). Loss %.3f.' % (
                epoch, step, globalstep, input_time, train_time, frcnn_time, loss))

    if step % global_config.cfg['summary_interval'] == 0:
        train_writer.add_summary(summary, globalstep)

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


def generate_cls_gt(input_pipeline_out, frcnn_out):
    target_bbs = input_pipeline_out['target_bbs']

    batch_size = target_bbs.shape[0]
    sequence_size = target_bbs.shape[1]
    PLAYERS = 10

    # The groundtruth bbs with shape (batch_size, backprop_step_size, PLAYERS=10, 4).
    target_bbs = np.reshape(target_bbs, (batch_size, sequence_size, PLAYERS, 4))

    # The RPs with shape (batch_size, sequence_length, PLAYERS=10, 4).
    rp_bbs = frcnn_out

    # The groundtruth for the classifier with shape ().
    # Each RP bb is one of the following classes: t from [0, 1, ..., PLAYERS=10].
    # 0 means take LSTM prediction.
    target_cls = np.zeros((batch_size, sequence_size, PLAYERS))

    threshold = 0.8

    for b in range(batch_size):
        for s in range(sequence_size):
            # Shape (10, 4).
            current_gt_bbs = target_bbs[b, s]

            # Shape (10, 4).
            current_rp_bbs = rp_bbs[b, s]

            # Iterate through all rp bbs in this image.
            for i in range(PLAYERS):
                # Shape (4) in format [ymin, xmin, ymax, xmax].
                current_rp_bb = current_rp_bbs[i]

                ious = np.zeros(PLAYERS)

                # Iterate through all gt bbs in this image.
                for j in range(PLAYERS):
                    # Shape (4) in format [x, y, w, h].
                    current_gt_bb = current_gt_bbs[j]

                    ious[j] = util.helper.iou(current_gt_bb,
                                              util.helper.ymin_xmin_ymax_xmax_to_xywh(current_rp_bb))

                index = np.argmax(ious)
                if ious[index] >= threshold:
                    target_cls[b, s, i] = index + 1
                else:
                    target_cls[b, s, i] = 0

    return target_cls


def run(sess, tensors, input_handles, train_writer, epoch, saver, globalstep, frcnn, validate=True):
    """
    tensors is a dict with the following keys:

      'input_data': {
          'groundtruth_bbs': groundtruth_bbs,
          'target_bbs': target_bbs,
          'images': images
      }

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

      'summary': summary.

    input_handles is a dict:

      input_handles = {
          'training_initializer': input_handles['tr_it'].initializer,
          'validation_initializer': input_handles['val_it'].initializer,
          'training_handle': training_handle,
          'validation_handle': validation_handle,
          'handle': input_handles['h']
      }
    """
    sess.run(input_handles['training_initializer'])

    step = 0

    # Go through one epoch.
    while True:
        try:
            input_pipeline_out, input_time = read_input(sess, tensors, input_handles)

            frcnn_out, frcnn_time = predict_frcnn(input_pipeline_out['images'], frcnn)

            target_cls = generate_cls_gt(input_pipeline_out, frcnn_out)

            pass

            util.helper.draw_bb_and_cls_labels_and_save(input_pipeline_out['images'][0, 0],
                                                        frcnn_out[0, 0],
                                                        input_pipeline_out['target_bbs'][0, 0],
                                                        target_cls[0, 0],
                                                        '0_0')

            # lstm_out, summary_out, train_time = train(sess, tensors, input_pipeline_out, frcnn_out)
            #
            # interval_actions(epoch, step, globalstep,
            #                  input_time, train_time, 0,
            #                  lstm_out['total_loss'], train_writer, summary_out)

            step += 1
            globalstep += 1
        except tf.errors.OutOfRangeError:
            break

    return globalstep
