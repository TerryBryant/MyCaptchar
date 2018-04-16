## 基于tensorflow框架的CNN多标签分类识别
首先介绍整体思路：由于手工标注的数据集十分有限，所以打算用finetune的方法，具体流程如下  
1、用captcha自动生成六万多张假的验证码图片进行训练，得到模型文件  
2、根据catpcha训练得到的模型文件，固定前面的卷积层，将后面的全连接层拿来进行finetune，得到最终的模型文件  
3、编写inference文件，进行预测  
### 下面逐一介绍每个文件
1、```yzm_annotate```该文件夹下是一个基于python-qt写的人工标注工具，真实的验证码标注全通过它来完成


2、```train.py```该文件是最初的训练代码，参考的是网上的教程，只考虑实现的逻辑，训练起来特别慢。采用的网络结构是一个三层的CNN，将验证码
转成数字形式，再进行one-hot编码，由此来训练，整个结构还是很简单的


3、```train_set_gen.py```该文件用于生成六万多张图片，并转换为灰度图进行保存（彩图对cnn意义不大），用于后面的训练


4、```make_tfrecords.py```该文件用于将数据集制作成train.tfrecords和valid.tfrecords，有了tfrecords文件，训练速度大大提升


5、```train_estimator.py```该文件采用estimator+tf.data.Dataset这些高级api来进行训练，代码更清晰，训练速度也很快，在我的1050Ti
显卡上，三层CNN，66x34大小的图片，一秒钟能跑3000多张图


6、```train_finetune.py```该文件用于finetune，将六万多张自动生成的图片训练得到的模型，拿来finetune，这里用的是estimator的WarmStartSettings，
tensorflow1.6版本才开始支持的，官网教程也是十几天前才出的。。


7、```ckpt2pb.py```该文件用于将ckpt模型文件转化为pb文件，即包含参数名和参数值的模型文件，下面inference会用到


8、```inference.py```该文件用于inference。说来惭愧，这个事情前前后后拖了两个月才完结，搞得现在的验证码跟当时的验证码都不太一样了，
不过幸好只是尺寸变化了，resize一下还是能用的，最后的效果也还不错

最终，由于真实用来训练的验证码只有七百多张，所以准确率只达到了88%（不做finetune直接训练这七百多张，准确率只能到百分之四五十，所以折腾了这么久还是有点用的）。容易出错的是1和7，再就是这个[c和e](https://github.com/TerryBryant/MyCaptchar/blob/master/cnn_tensorflow/res_image/2ec8.png)，实在没办法，毕竟训练集太有限
### 补充说明
看看最后的识别结果:

![识别结果](https://github.com/TerryBryant/MyCaptchar/blob/master/cnn_tensorflow/res_image/%E8%AF%86%E5%88%AB%E7%BB%93%E6%9E%9C.PNG)
