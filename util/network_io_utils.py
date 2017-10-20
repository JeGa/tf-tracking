import numpy as np

import util.helper


def cls_pred(predictions, batch, time):
    classifications = np.zeros((10, 11))
    for i in range(len(predictions)):
        classifications[i] = predictions[i][batch, time]
    return classifications


def cls_pred_final(predictions, batch, time):
    return np.squeeze(np.argmax(cls_pred(predictions, batch, time), axis=1))


def read_input(sess, tensors, input_handles, dataset):
    out_input_pipeline = None

    with util.helper.timeit() as input_time:
        # Images are normalized.
        if dataset == 'training':
            handle = input_handles['training_handle']
        elif dataset == 'validation':
            handle = input_handles['validation_handle']
        else:
            raise ValueError('Specify valid dataset.')

        out_input_pipeline = sess.run(tensors, feed_dict={input_handles['handle']: handle})

    return out_input_pipeline, input_time.time()


def sort_rps(frcnn_out, ordered_last_region_proposals,
             cls_predictions,
             ordered_lstmreg_input,
             reg_predictions,
             t, target_cls_init):
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
        current_cls_prediction = cls_pred_final(cls_predictions, j, t)

        if t == 0:
            # TODO
            frcnn_out_ordered = frcnn_out[j, t]
            frcnn_out_unordered = frcnn_out[j, t]

            for i in range(10):
                tcl = int(target_cls_init[j, i])
                if tcl != 0:
                    frcnn_out_ordered[i] = frcnn_out_unordered[tcl - 1]

            ordered_last_region_proposals[j, t + 1] = frcnn_out_ordered
            ordered_lstmreg_input[j, t + 1] = frcnn_out_ordered

            # ordered_last_region_proposals[j, t + 1] = frcnn_out[j, t]
            # ordered_lstmreg_input[j, t + 1] = frcnn_out[j, t]
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


def predict_frcnn(sequence_images, frcnn):
    """
    Uses an own session for the rcnn graph.
    """
    batch_size = sequence_images.shape[0]
    sequence_length = sequence_images.shape[1]

    frcnn_out = np.zeros((batch_size,
                          sequence_length,
                          10, 4))

    with util.helper.timeit() as frcnn_time:
        for i in range(batch_size):
            for j in range(sequence_length):
                bb, _ = frcnn.predict((np.expand_dims(sequence_images[i][j], 0) * 255).astype(np.uint8))
                for k in range(10):
                    frcnn_out[i][j][k] = util.helper.ymin_xmin_ymax_xmax_to_xywh(bb[k])

    return frcnn_out, frcnn_time.time()
