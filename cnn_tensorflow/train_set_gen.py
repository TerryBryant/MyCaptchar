from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import cv2
import random
import os
import skimage.io as IO


# 验证码中的字符
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f']


# 长度为四个字符，不区分大小写
def random_captcha_text(char_set=number+alphabet, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


# 生成字符对应的验证码
def gen_captcha_text_and_image(image_width, image_height):
    image = ImageCaptcha()

    captcha_text = random_captcha_text()
    captcha_text = "".join(captcha_text)

    captcha = image.generate(captcha_text)
    captcha_image = IO.imread(captcha)

    # captcha_image = cv2.imread(captcha_image)
    captcha_image = cv2.cvtColor(captcha_image, cv2.COLOR_BGR2GRAY)
    captcha_image = cv2.resize(captcha_image, (image_width, image_height))
    return captcha_text, captcha_image



if __name__ == "__main__":
    num = 100000
    while num > 0:
        text, image = gen_captcha_text_and_image(image_width=66, image_height=34)
        img_name = os.path.join("captcha", text + ".png")
        cv2.imwrite(img_name, image)

        print(num)
        num -= 1
