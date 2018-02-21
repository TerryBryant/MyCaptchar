import tensorflow as tf
import os
import random
import skimage.io as IO
import numpy as np


IMAGE_HEIGHT = 34
IMAGE_WIDTH = 66
MAX_CAPTCHA = 4
# CHAR_SET_LEN = 6

train_set = "ss/"
test_set = "ss_test/"
# 验证码中的字符
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f']

char_set = number + alphabet
CHAR_SET_LEN = len(char_set)


def get_name_and_image():
    all_image = os.listdir(train_set)
    random_file = random.randint(0, len(all_image)-1)
    name = all_image[random_file].split(".")[0]
    image = IO.imread(os.path.join(train_set, all_image[random_file]))
    return name, image


# 文本转向量
def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        print("标注有误：", text)
        raise ValueError("验证码最长4个字符")

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    def char2pos(c):
        k = ord(c) - ord('0')   # 从'0'开始计数
        if k > 9:
            k = ord(c) - ord('a') + 10  # '0'到'9'之后开始计10
        return k

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1

    return vector

# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        else:
            char_code = char_idx + ord('a') - 10
        text.append(chr(char_code))

    return "".join(text)

# if __name__ == "__main__":
#     name = text2vec("12ac")
#     vec = vec2text(name)
#     exit()

# 生成一个训练batch
def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    for i in range(batch_size):
        name, image = get_name_and_image()
        batch_x[i, :] = (image.flatten() - 128) / 128 # 标准化
        batch_y[i, :] = text2vec(name)
    return batch_x, batch_y

# 开始构建cnn
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32) # dropout

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

def crack_captchar_cnn(X, keep_prob=0.5):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    w_c1 = weight_variable([3, 3, 1, 16], name='w_c1')
    b_c1= weight_variable([16], name='b_c1')
    conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x, w_c1), b_c1))
    conv1 = max_pool_2x2(conv1, name='pool1')
    # conv1 = tf.nn.dropout(conv1, keep_prob=keep_prob)

    w_c2 = weight_variable([3, 3, 16, 32], name='w_c2')
    b_c2 = weight_variable([32], name='b_c2')
    conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(conv1, w_c2), b_c2))
    conv2 = max_pool_2x2(conv2, name='pool2')
    # conv2 = tf.nn.dropout(conv2, keep_prob=keep_prob)

    # fully connected layer
    w_d = weight_variable([9 * 17 * 32, 1024], name='w_d')
    b_d = weight_variable([1024], name='b_d')
    dense = tf.reshape(conv2, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    # dense = tf.nn.dropout(dense, keep_prob=keep_prob)

    w_out = weight_variable([1024, MAX_CAPTCHA * CHAR_SET_LEN], name='w_out')
    b_out = weight_variable([MAX_CAPTCHA * CHAR_SET_LEN], name='b_out')
    out = tf.add(tf.matmul(dense, w_out), b_out)

    return out

def train_crack_captcha_cnn():
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
            batch_x, batch_y = get_next_batch(100)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x,
                                                              Y: batch_y,
                                                              keep_prob: 0.75})

            if step % 100 == 0:
                print("step:%s, loss: %s" % (step, loss_))

            # count accuracy every 100 steps
            if step % 1000 == 0:
                batch_x_test, batch_y_test = get_next_batch(18)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test,
                                                    Y: batch_y_test,
                                                    keep_prob: 1.})
                print("printing test accuracy...")
                print("step:%s, test acc: %s" % (step, acc))

                if step % 5000 == 0:
                    saver.save(sess, "trained_model/captchar.model", global_step=step)

            step += 1


if __name__ == "__main__":
    train_crack_captcha_cnn()
