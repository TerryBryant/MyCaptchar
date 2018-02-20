import os
import cv2
import numpy as np


import matplotlib.pyplot as plt
# change origin captchar to binary image
for file in os.listdir("ss"):
    img_path = os.path.join("ss", file)
    img = cv2.imread(img_path)
    # BGR转HSV
    img_h, img_w = img.shape[:2]
    img_bi = np.zeros([img_h, img_w], dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for i in range(img_h):
        for j in range(img_w):
            if img[i, j, 0] > 0 and img[i, j, 1] > 160:
                img_bi[i, j] = 255

    img_floodfill = img_bi.copy()
    # mask used to flood filling, notice the size needs to be 2 pixels more than the origin image
    mask = np.zeros((img_h + 2, img_w + 2), np.uint8)
    # floodfill from point(0, 0)
    cv2.floodFill(img_floodfill, mask, (0, 0), 255)
    # invert floodfilled image
    img_floodfill = cv2.bitwise_not(img_floodfill)
    # combine the two images to get the foreground
    img_bi_filled = img_bi | img_floodfill

    # find the minimum enclosing rectangles, to locate four letters' horizontal position
    im, contours, hierarchy = cv2.findContours(img_bi_filled, 1, 2)
    bound_list = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 19:
            bound_list.append(x)
            bound_list.append(w)
        elif w < 31:
            bound_list.append(x)
            bound_list.append(w/2)
            bound_list.append(x+w/2)
            bound_list.append(w/2)
        elif w < 41:
            bound_list.append(x)
            bound_list.append(w/3)
            bound_list.append(x+w/3)
            bound_list.append(w/3)
            bound_list.append(x+w*2/3)
            bound_list.append(w/3)
        else:
            bound_list.append(x)
            bound_list.append(w/4)
            bound_list.append(x+w/4)
            bound_list.append(w/4)
            bound_list.append(x+w*2/4)
            bound_list.append(w/4)
            bound_list.append(x+w*3/4)
            bound_list.append(w/4)

    # find the vertical locations of the four letters
    bound_above = 0
    bound_bottom = 0
    vert_proj = np.sum(img_bi, axis=1)
    for i in range(len(vert_proj)):
        if vert_proj[i] > 0:
            bound_above = i
            break
    for i in range(len(vert_proj)-1, 0, -1):
        if vert_proj[i] > 0:
            bound_bottom = i
            break

    # # todo list...
    # # 四个字符的numpy数据
    # for i in range(0, len(bound_list), 2):
    #     img_single_letter = img_bi[bound_above:bound_bottom, int(bound_list[i]):int(bound_list[i]+bound_list[i+1])]
    #     # ...pytesser3 recognize


    # for display
    img_show = cv2.cvtColor(img_bi, cv2.COLOR_GRAY2BGR)
    for i in range(0, len(bound_list), 2):
        cv2.rectangle(img_show, (int(bound_list[i]), bound_above), (int(bound_list[i]+bound_list[i+1]), bound_bottom), (0, 0, 255))
    cv2.imshow("chaoye", img_bi)
    cv2.imshow("chaoye", img_show)
    cv2.waitKey(0)
exit()
