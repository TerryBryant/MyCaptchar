import tensorflow as tf
import numpy as np
import os
import skimage.io as io

IMAGE_HEIGHT = 34
IMAGE_WIDTH = 66
MAX_CAPTCHA = 4    # 处理四位字符的验证码


# 验证码中的字符
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f']

char_set = number + alphabet
CHAR_SET_LEN = len(char_set)


# 文本转向量
def text2int(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        print("标注有误：", text)
        raise ValueError("验证码最长4个字符")

    labels = [0] * text_len
    for i in range(text_len):
        k = ord(text[i]) - ord('0')      # 从'0'开始计数
        if k > 9:
            k = ord(text[i]) - ord('a') + 10     # '0'到'9'之后开始计10

        labels[i] = k

    return labels


def get_file(file_dir):
    images = []
    labels = []

    for file in os.listdir(file_dir):
        images.append(os.path.join(file_dir, file))
        labels.append(file.split(".")[0])

    # 这里的shuffle不一定需要，因为在后面训练时，tf.data有shuffle功能
    temp = np.array([images, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])

    return image_list, label_list


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord(images, labels, save_dir, name):
    filename = os.path.join(save_dir, name + ".tfrecords")
    n_samples = len(labels)

    if len(images) != n_samples:
        raise ValueError('Images size %d does not match label size %d' % (len(images), n_samples))

    writer = tf.python_io.TFRecordWriter(filename)
    print("\nTransform start...")

    for i in range(n_samples):
        try:
            image = io.imread(images[i])
            image_raw = image.tostring()
            label = text2int(labels[i])
            example = tf.train.Example(features=tf.train.Features(feature={
                'label0': int64_feature(label[0]),
                'label1': int64_feature(label[1]),
                'label2': int64_feature(label[2]),
                'label3': int64_feature(label[3]),
                'image_raw': bytes_feature(image_raw)
            }))

            writer.write(example.SerializeToString())
        except IOError as e:
            print("Could not read ", images[i])
            print("error: ", e)
            print("Skip it!\n")
    writer.close()
    print("Transform done!")


test_dir = "captcha/"
images, labels = get_file(test_dir)

save_dir = "./"
name = "captcha_train"
convert_to_tfrecord(images, labels, save_dir, name)

