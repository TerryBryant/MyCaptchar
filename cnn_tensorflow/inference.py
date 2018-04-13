import cv2
import numpy as np
import time
import tensorflow as tf

IMAGE_HEIGHT = 34
IMAGE_WIDTH = 66

# 验证码中的字符
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f']

char_set = number + alphabet

img_path = "ss_test/c019.png"
img_name = img_path.split("/")[1].split(".")[0]
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_h, img_w = img.shape[:2]
img_bi = np.zeros([img_h, img_w], dtype=np.uint8)

for i in range(img_h):
    for j in range(img_w):
        if img[i, j, 0] > 0 and img[i, j, 1] > 160:
            img_bi[i, j] = 255

img_bi = cv2.resize(img_bi, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_CUBIC)  # 两个月过去了，验证码的尺寸都变了。。。
img_bi = img_bi.astype(np.float)
img_bi = (img_bi - 128) / 128.0
input_blob = np.reshape(img_bi, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])



# read model file and inference
t1 = time.time()
with open("trained_model/frozen_model.pb", "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name="")

    with tf.Session() as sess:
        # 从pb文件中读取变量名
        data = sess.graph.get_tensor_by_name("x_input:0")
        predict = sess.graph.get_tensor_by_name("x_predict:0")


        # 开始预测
        sess.run(tf.global_variables_initializer())
        img_out = sess.run(predict, feed_dict={data: input_blob})

        # 转化成英文字母
        out_str = ""
        max_idx_p = np.argmax(img_out, 2)
        for i in range(max_idx_p.shape[1]):
            tmp_str = char_set[max_idx_p[0, i]]
            out_str += tmp_str

        print("真实值：", img_name)
        print("预测值：", out_str)

t2 = time.time()
print("所花时间： ", t2 - t1)
exit()
