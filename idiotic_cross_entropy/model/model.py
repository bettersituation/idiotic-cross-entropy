from idiotic_cross_entropy.model.net import Net
from idiotic_cross_entropy.model.board import Board
import tensorflow as tf


class Model(Net, Board):
    def __init__(self, sess, feature_shape, label_num, idiotic_alpha, train_loss='cross_entropy', board_path=None):
        self._sess = sess
        Net.__init__(self, sess, feature_shape, label_num, idiotic_alpha, train_loss)
        Board.__init__(self, board_path)

    def reset(self):
        self._sess.run(tf.global_variables_initializer())
