import tensorflow as tf
import numpy as np
import cv2

MAX_CAPTCHA = 4
# 验证码中的字符
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f']

char_set = number + alphabet
CHAR_SET_LEN = len(char_set)


# read and convert the image
img_name = "ss_origin/cfd9.png"
img = cv2.imread(img_name)


# BGR转HSV
img_h, img_w = img.shape[:2]
img_bi = np.zeros([img_h + 1, img_w + 1], dtype=np.uint8)  # 得到偶数长宽的图片
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
for i in range(img_h):
    for j in range(img_w):
        if img[i, j, 0] > 0 and img[i, j, 1] > 160:
            img_bi[i, j] = 255

img_bi = img_bi.astype(np.float)
img_bi = (img_bi - 128) / 128
input_blob = np.reshape(img_bi, [-1, img_h+1, img_w+1, 1])

# read model file and inference
with open("trained_model/frozen_model_10000.pb", "rb") as f:
    out_graph_def = tf.GraphDef()
    out_graph_def.ParseFromString(f.read())
    tf.import_graph_def(out_graph_def, name="")

    with tf.Session() as sess:
        # 从pb文件中读取变量名
        data = sess.graph.get_tensor_by_name("x_input:0")
        predict = sess.graph.get_tensor_by_name("x_predict:0")
        keep_prob = sess.graph.get_tensor_by_name("keep_prob:0")

        # 开始预测
        sess.run(tf.global_variables_initializer())
        img_out = sess.run(predict, feed_dict={data: input_blob, keep_prob: 1.0})

        # 转化成英文字母
        out_str = ""
        max_idx_p = np.argmax(img_out, 2)
        for i in range(max_idx_p.shape[1]):
            tmp_str = char_set[max_idx_p[0, i]]
            out_str += tmp_str

        print("真实值：", img_name.split("/")[-1].split(".")[0])
        print("预测值：", out_str)
