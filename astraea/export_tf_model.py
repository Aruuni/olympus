import tensorflow as tf

graph = tf.Graph()
with graph.as_default():
    saver = tf.train.import_meta_graph("/home/xudong/astraea/tmp/model.meta")
    with tf.Session() as sess:
        saver.restore(sess, '/home/xudong/astraea/tmp/model')
        global_step = tf.get_collection('global_step')[0]
        tf.get_default_graph().clear_collection('global_step')
        tf.get_default_graph().as_graph_def().node.remove(global_step.op.node_def)
        saver.save(sess, "/home/xudong/astraea/tmp/model")