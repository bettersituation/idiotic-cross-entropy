import tensorflow as tf


class Net:
    def __init__(self, sess, feature_shape, label_num, idiotic_alpha, train_loss='cross_entropy'):
        self._sess = sess
        self._idiotic_alpha = idiotic_alpha
        self._train_loss = train_loss

        self.x_ph = tf.placeholder(tf.float32, [None, *feature_shape], 'x')
        self.y_ph = tf.placeholder(tf.float32, [None, label_num], 'y')

        self.l1 = tf.layers.dense(self.x_ph, 256, tf.nn.relu)
        self.l2 = tf.layers.dense(self.l1, 256, tf.nn.relu)
        self.pred_y = tf.layers.dense(self.l2, label_num, tf.nn.softmax)

        self.idiotic_label = tf.stop_gradient((1 - self._idiotic_alpha) * self.y_ph + self._idiotic_alpha * self.pred_y)

        self.train_cross_entropy_loss = - tf.reduce_mean(tf.reduce_sum(self.idiotic_label * tf.log(self.pred_y + 1e-8), axis=1))
        self.train_l1_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(self.idiotic_label - self.pred_y), axis=1))
        self.train_l2_loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.idiotic_label, self.pred_y), axis=1))

        self.real_cross_entropy_loss = - tf.reduce_mean(tf.reduce_sum(self.y_ph * tf.log(self.pred_y + 1e-8), axis=1))
        self.real_l1_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(self.y_ph - self.pred_y), axis=1))
        self.real_l2_loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.y_ph, self.pred_y), axis=1))

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.pred_y, 1), tf.argmax(self.y_ph, 1)), tf.float32))

        self.training_step = tf.train.get_or_create_global_step()

        opt = tf.train.AdamOptimizer()
        if self._train_loss == 'cross_entropy':
            self.train_op = opt.minimize(self.train_cross_entropy_loss, global_step=self.training_step)
        elif self._train_loss == 'l1':
            self.train_op = opt.minimize(self.train_l1_loss, global_step=self.training_step)
        elif self._train_loss == 'l2':
            self.train_op = opt.minimize(self.train_l2_loss, global_step=self.training_step)

    def train(self, x, y):
        training_step, cross_entropy_loss, l1_loss, l2_loss, accuracy, _ = self._sess.run([self.training_step, self.real_cross_entropy_loss, self.real_l1_loss, self.real_l2_loss, self.accuracy, self.train_op],
                                                                                          feed_dict={self.x_ph: x, self.y_ph: y}
                                                                                          )
        return training_step, cross_entropy_loss, l1_loss, l2_loss, accuracy

    def test(self, x, y):
        training_step, cross_entropy_loss, l1_loss, l2_loss, accuracy = self._sess.run([self.training_step, self.real_cross_entropy_loss, self.real_l1_loss, self.real_l2_loss, self.accuracy],
                                                                                       feed_dict={self.x_ph: x, self.y_ph: y}
                                                                                       )
        return training_step, cross_entropy_loss, l1_loss, l2_loss, accuracy
