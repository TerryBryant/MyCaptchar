import tensorflow as tf
import numpy as np
import cv2

MAX_CAPTCHA = 4
CHAR_SET_LEN = 16


# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        else:
            char_code = char_idx + ord('a') - 10
        text.append(chr(char_code))

    return "".join(text)


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

img_bi = img_bi.astype(np.float32)
img_bi = (img_bi - 128) / 128


# input_blob = cv2.dnn.blobFromImage(img_bi)
# net = cv2.dnn.readNetFromTensorflow("trained_model/frozen_model_10000.pb")
# net.setInput(input_blob)
# result = net.forward()
# print(result)
#
#
#
# exit()

input_blob = np.reshape(img_bi, [-1, (img_h+1) * (img_w+1)])

# read model file and inference
with open("trained_model/frozen_model_10000.pb", "rb") as f:
    out_graph_def = tf.GraphDef()
    out_graph_def.ParseFromString(f.read())
    tf.import_graph_def(out_graph_def, name="")

    with tf.Session() as sess:
        data = sess.graph.get_tensor_by_name("input/x_input:0")
        predict = sess.graph.get_tensor_by_name("softmax/predict:0")

        # print(data)
        # print(predict)
        # print("\n")
        # print(input_blob.shape)
        # print(input_blob.dtype)
        # exit()

        sess.run(tf.global_variables_initializer())
        img_out = sess.run(predict, feed_dict={data: input_blob})
        print(img_out)
        exit()

        # 转化成英文字母
        out_str = ""
        max_idx_p = tf.argmax(img_out, 2)
        for i in range(max_idx_p.shape[1]):
            tmp_str = vec2text(img_out[i])
            # tmp_str = vec2text(img_out(max_idx_p[i]))
            out_str.append(tmp_str)

        print(out_str)


