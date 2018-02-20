import tensorflow as tf
import os
import random
import cv2
import numpy as np


IMAGE_HEIGHT = 33
IMAGE_WIDTH = 65
MAX_CAPTCHA = 4
CHAR_SET_LEN = 26


def get_name_and_image():
    all_image = os.listdir("ss2/")
    random_file = random.randint(0, 800)
    base = os.path.basename("ss2/" + all_image[random_file])

    name = os.path.splitext(base)[0]
    image = cv2.imread(os.path.join("ss2/", all_image[random_file]))
    return name, image


def name2vec(name):
    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    for i, c in enumerate(name):
        idx = i * 26 + ord(c) - 97
        vector[idx] = 1
    return vector

def vec2name(vec):
    name = []
    for i in vec:
        a = chr(i + 97)
        name.append(a)
    return "".join(name)

def get_next_batch(batch_size=64):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    for i in range(batch_size):
        name, image = get_name_and_image()
        batch_x[i, :] = 1 * (image.flatten())
        batch_y[i, :] = name2vec(name)
    return batch_x, batch_y

def crack_captchar_cnn(X, w_alpha=0.01, b_alpha=0.1, keep_prob=0.5):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    w_c1 = tf.Variable(w_alpha * tf.random_normal(5, 5, 1, 32))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding="SAME"), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    conv1 = tf.nn.dropout(conv1, keep_prob=keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal(5, 5, 1, 64))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding="SAME"), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    conv2 = tf.nn.dropout(conv2, keep_prob=keep_prob)

    # fully connected layer
    w_d = tf.Variable(w_alpha * tf.random_normal([15 * 57 * 64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv2, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob=keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)

    return out

def train_crack_captcha_cnn(X, Y):
    output = crack_captchar_cnn(X)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.arg_max(predict, 2)
    max_idx_l = tf.arg_max(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x, batch_y = get_next_batch(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x,
                                                              Y: batch_y,
                                                              keep_prob: 0.75})
            print("step:%s, loss: %s" % (step, loss_))

            # count accuracy every 100 steps
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(21)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test,
                                                    Y: batch_y_test,
                                                    keep_prob: 1.})
                print("printing test accuracy...")
                print("step:%s, loss: %s" % (step, acc))

                if acc > 0.9:
                    saver.save(sess, "captchar.model", global_step=step)
                    break
            step += 1


X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32) # dropout

if __name__ == "__main__":
    train_crack_captcha_cnn(X, Y, )
