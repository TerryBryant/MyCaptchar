#coding=utf8
from __future__ import print_function
import tensorflow as tf
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 训练相关超参数
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 100


# 验证码相关信息
IMAGE_HEIGHT = 34
IMAGE_WIDTH = 66
MAX_CAPTCHA = 4
IMAGE_CHANNELS = 3

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f']

char_set = number + alphabet
CHAR_SET_LEN = len(char_set)


# 构建深度学习网络
def weight_variable(shape, name):
    initial = tf.random_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x, name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def training_network(X):
    with tf.variable_scope("input"):
        x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name="x_input")

    with tf.variable_scope("conv1"):
        w_c1 = weight_variable([3, 3, 1, 32], name='w_c1')
        b_c1= weight_variable([32], name='b_c1')
        conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x, w_c1), b_c1))
        conv1 = max_pool_2x2(conv1, name='pool1')
        conv1 = tf.nn.dropout(conv1, keep_prob=keep_prob)

    with tf.variable_scope("conv2"):
        w_c2 = weight_variable([3, 3, 32, 64], name='w_c2')
        b_c2 = weight_variable([64], name='b_c2')
        conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(conv1, w_c2), b_c2))
        conv2 = max_pool_2x2(conv2, name='pool2')
        conv2 = tf.nn.dropout(conv2, keep_prob=keep_prob)

    # fully connected layer
    with tf.variable_scope("fc"):
        w_d = weight_variable([9 * 17 * 64, 1024], name='w_d')
        b_d = weight_variable([1024], name='b_d')
        dense = tf.reshape(conv2, [-1, w_d.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
        dense = tf.nn.dropout(dense, keep_prob=keep_prob)

    with tf.variable_scope("softmax"):
        w_out = weight_variable([1024, MAX_CAPTCHA * CHAR_SET_LEN], name='w_out')
        b_out = weight_variable([MAX_CAPTCHA * CHAR_SET_LEN], name='b_out')
        out = tf.add(tf.matmul(dense, w_out), b_out)

    return out


# 用于解析tfrecords数据
def _parse_function(proto):
    features = {'label0': tf.FixedLenFeature([1], tf.int64),
                'label1': tf.FixedLenFeature([1], tf.int64),
                'label2': tf.FixedLenFeature([1], tf.int64),
                'label3': tf.FixedLenFeature([1], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string, default_value='')}

    parsed_feature = tf.parse_single_example(proto, features)
    image = tf.image.decode_image(parsed_feature['image_raw'], channels=IMAGE_CHANNELS)
    # image = parsed_feature['image_raw']


    # label = tf.expand_dims(tf.constant([parsed_feature['label0'], parsed_feature['label1'],
    #                                     parsed_feature['label2'], parsed_feature['label3']]), 1)
    # index = tf.expand_dims(tf.range(MAX_CAPTCHA), 1)
    # concated = tf.concat([index, label], axis=1)
    # image_label = tf.sparse_to_dense(concated, [MAX_CAPTCHA, CHAR_SET_LEN], 1, 0)


    image_label0 = tf.sparse_to_dense(parsed_feature['label0'], [MAX_CAPTCHA, CHAR_SET_LEN], 1, 0)
    image_label1 = tf.sparse_to_dense(parsed_feature['label1'], [MAX_CAPTCHA, CHAR_SET_LEN], 1, 0)
    image_label2 = tf.sparse_to_dense(parsed_feature['label2'], [MAX_CAPTCHA, CHAR_SET_LEN], 1, 0)
    image_label3 = tf.sparse_to_dense(parsed_feature['label3'], [MAX_CAPTCHA, CHAR_SET_LEN], 1, 0)
    image_label = tf.concat([image_label0, image_label1, image_label2, image_label3], axis=1)

    return image, image_label


def train_captcha_cnn():
    # 读取tfrecords数据
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function, num_parallel_calls=16).repeat(EPOCHS).batch(BATCH_SIZE)
    iter = dataset.make_initializable_iterator()
    X, Y = iter.get_next()

    # # 训练相关函数
    # output = training_network(X)    # 得到输出结果
    # with tf.variable_scope("loss"):
    #     loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output[:, 0:16], labels=Y[:, 0:16]))
    #     loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output[:, 16:32], labels=Y[:, 16:32]))
    #     loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output[:, 32:48], labels=Y[:, 32:48]))
    #     loss4 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output[:, 48:], labels=Y[:, 48:]))
    #     loss = (loss1 + loss2 + loss3 + loss4) / 4.0
    #
    # with tf.variable_scope("train"):
    #     optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    #
    # with tf.variable_scope("accuracy"):
    #     predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN], name="x_predict")
    #     max_idx_p = tf.argmax(predict, 2)
    #     max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    #
    #     correct_pred = tf.equal(max_idx_p, max_idx_l)
    #     accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    # 开始训练
    output_node_names = "x_input,x_predict,keep_prob"
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 训练EPOCHS轮
        global_step = 0
        for epoch in range(EPOCHS):
            sess.run(iter.initializer, feed_dict={filenames: train_set})
            while True:
                try:
                    global_step += 1
                    sess.run([X, Y])

                    print(Y)
                    exit()
                    #_, loss_ = sess.run([optimizer, loss], feed_dict={keep_prob: 0.75})
                    #print("当前epoch：%d，训练步数：%d，损失率：%f" % (epoch, global_step, loss_))
                except tf.errors.OutOfRangeError:
                    break

            # 一个epoch跑完，计算此时的验证集的准确率
            sess.run(iter.initializer, feed_dict={filenames: valid_set})
            sess.run([X, Y])
            #acc = sess.run(accuracy, feed_dict={keep_prob: 1.})
            #print("当前epoch：%d，测试集准确率：%f" % (epoch, acc))

            # 每个epoch保存一次模型
            constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                          output_node_names.split(','))
            filename = "trained_model/frozen_model_%s.pb" % global_step
            with tf.gfile.GFile(filename, "wb") as f:
                f.write(constant_graph.SerializeToString())


if __name__ == "__main__":
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # dropout
    filenames = tf.placeholder(tf.string, shape=None)     # tfrecorfds的文件名

    # tfrecords file name
    train_set = tf.Variable("captcha_train.tfrecords", dtype=tf.string)
    valid_set = tf.Variable("captcha_valid.tfrecords", dtype=tf.string)
    # train_set = "captcha_train.tfrecords"
    # valid_set = "captcha_valid.tfrecords"
    train_captcha_cnn()