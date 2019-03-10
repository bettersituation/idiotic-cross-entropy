from idiotic_cross_entropy.data.loader import Loader
from idiotic_cross_entropy.model.model import Model
from idiotic_cross_entropy.config import *
import tensorflow as tf


if __name__ == '__main__':
    sess = tf.Session()
    idiotic_alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    board_path = '../.out/'
    batch_size = [128]
    data_types = ['mnist', 'cifar10', 'cifar100', 'fashion_mnist']
    train_losses_type = ['cross_entropy', 'l1', 'l2']

    for data_type in data_types:
        data_loader = Loader(data_type)
        test_feature, test_label = data_loader.get_test_batch(flatten=False)
        feature_shape, label_num = data_loader.get_shape(flatten=False)
        if len(feature_shape) == 2:
            feature_shape.append(1)

        for batch_s in batch_size:
            for train_loss_type in train_losses_type:
                for idiotic_alpha in idiotic_alphas:

                    model = Model(sess, feature_shape, label_num, idiotic_alpha, TRAIN_LOSS,
                                  board_path + '{data}/{loss}_loss/batch_size_{batch}/alpha_{alpha}'.format(
                                      data=data_type, loss=train_loss_type, batch=batch_s, alpha=idiotic_alpha)
                                  )
                    model.reset()

                    for i in range(1000):
                        train_feature, train_label = data_loader.get_train_batch(BATCH_SIZE, flatten=False)
                        training_step, train_cross_entropy, train_l1, train_l2, train_accuracy = model.train(train_feature, train_label)

                        if i % 10 == 9:
                            train_summary_dict = {'train/cross_entropy': train_cross_entropy,
                                                  'train/l1': train_l1,
                                                  'train/l2': train_l2,
                                                  'train/accuracy': train_accuracy}
                            model.add_scalars(train_summary_dict, training_step, False)

                            test_step, test_cross_entropy, test_l1, test_l2, test_accuracy = model.test(test_feature, test_label)
                            test_summary_dict = {'test/cross_entropy': test_cross_entropy,
                                                 'test/l1': test_l1,
                                                 'test/l2': test_l2,
                                                 'test/accuracy': test_accuracy}
                            model.add_scalars(test_summary_dict, test_step, False)

                            if i % 200 == 199:
                                print('training step:{:4d}, cross entropy:{:6.4f}, l1 loss:{:6.4f}, l2 loss:{:6.4f}, accuracy: {:4.2f}%'.format(training_step, train_cross_entropy, train_l1, train_l2, train_accuracy*100))
                                print('test step:{:4d}, cross entropy:{:6.4f}, l1 loss:{:6.4f}, l2 loss:{:6.4f} accuracy: {:4.2f}%'.format(test_step, test_cross_entropy, test_l1, test_l2, test_accuracy*100))
                                model.flush()

                    model.flush()
                    del model
                    print('idiotic alpha: {}, batch size: {} experiment done'.format(idiotic_alpha, batch_s))
