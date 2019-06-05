# 本脚本用于产生tfrecords文件
# 输入训练集文件路径，并根据TRAIN_VAL_RATIO设置训练集和验证集的比例，
# 得到相应的train.tfrecords和val.tfrecords
# 本脚本用于产生tfrecords文件
# 输入训练集文件路径，并根据TRAIN_VAL_RATIO设置训练集和验证集的比例，
# 得到相应的train.tfrecords和val.tfrecords

# Warning in 2019/06/05
# 这个代码不适用于其它情况，因为没有考虑正负样本比例问题

import tensorflow as tf
import os

TRAIN_VAL_RATIO = 10  # 训练集与验证集的比例
MAX_CAPTCHA = 4  # 验证码长度为4

# # 验证码中的字符
# number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# alphabet = ['a', 'b', 'c', 'd', 'e', 'f']
#
# char_set = number + alphabet
# CHAR_SET_LEN = len(char_set)


# 将012a转成0、1、2、10
def text2int(text):
    text_len = len(text)
    if text_len != MAX_CAPTCHA:
        print("标注有误：", text)
        raise ValueError("验证码长度不为MAX_CAPTCHA个字符")

    labels = [0] * text_len
    for i in range(text_len):
        k = ord(text[i]) - ord('0')      # 从'0'开始计数
        if k > 9:
            k = ord(text[i]) - ord('a') + 10     # '0'到'9'之后开始计10

        labels[i] = k

    return labels

# 读取数据集相关信息
def get_file(file_dir):
    train_images = []
    train_labels = []

    val_images = []
    val_labels = []

    whole_files = os.listdir(file_dir)
    for i, file in enumerate(whole_files):
        if i % TRAIN_VAL_RATIO == 0:
            val_images.append(os.path.join(file_dir, file))
            val_labels.append(file.split(".")[0])  # 这里用文件名作为标注
        else:
            train_images.append(os.path.join(file_dir, file))
            train_labels.append(file.split(".")[0])

    # # 这里的shuffle不一定需要，因为在后面训练时，tf.data有shuffle功能
    # temp = np.array([images, labels])
    # temp = temp.transpose()
    # np.random.shuffle(temp)
    #
    # image_list = list(temp[:, 0])
    # label_list = list(temp[:, 1])

    return train_images, train_labels, val_images, val_labels


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord(images, labels, save_dir, name):
    filename = os.path.join(save_dir, name + ".tfrecords")
    num_samples = len(labels)

    if len(images) != num_samples:
        raise ValueError('Images size %d does not match label size %d' % (len(images), num_samples))

    writer = tf.python_io.TFRecordWriter(filename)
    print("Transform start...")

    num_real_samples = 0
    for i in range(num_samples):
        with tf.gfile.FastGFile(images[i], "rb") as f:
            image_encoded = f.read()
        label = text2int(labels[i])
        example = tf.train.Example(features=tf.train.Features(feature={
            'label0': int64_feature(label[0]),
            'label1': int64_feature(label[1]),
            'label2': int64_feature(label[2]),
            'label3': int64_feature(label[3]),
            'image_encoded': bytes_feature(image_encoded)
        }))

        writer.write(example.SerializeToString())
        num_real_samples += 1

    writer.close()
    print("Transform done!")
    return num_samples


if __name__ == "__main__":
    dataset_dir = "ss/"
    save_dir = "tfrecords_file/"
    train_file_name = "captcha_train"  # tfrecords文件名
    val_file_name = "captcha_valid"

    print("读取数据集文件中。。。")
    train_images, train_labels, val_images, val_labels = get_file(dataset_dir)
    print("制作train.tfrecords中。。。")
    num1_train = convert_to_tfrecord(train_images, train_labels, save_dir, train_file_name)
    print("制作val.tfrecords中。。。")
    num_val = convert_to_tfrecord(val_images, val_labels, save_dir, val_file_name)
    print("tfrecords文件制作完成，共有%d个训练集，%d个验证集" % (num1_train, num_val))

