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
