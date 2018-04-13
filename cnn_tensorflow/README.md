## 基于tensorflow框架的CNN多标签分类识别
首先介绍整体思路：由于手工标注的数据集十分有限，所以打算用finetune的方法，具体流程如下  
1、用captcha自动生成六万多张图片进行训练，得到模型文件  
2、根据catpcha训练得到的模型文件，固定前面的卷积层，将后面的全连接层拿来进行finetune，得到最终的模型文件  
3、编写inference文件，进行预测  
### 下面逐一介绍每个文件
```yzm_annotate```该文件夹下是一个基于python-qt写的人工标注工具，真实的验证码标注全通过它来完成  
```train.py```该文件是最初的训练代码，参考的是网上的教程，只考虑实现，未考虑训练效率  
```train_set_gen.py```该文件用于生成六万多张图片，并转换为灰度图进行保存（彩图对cnn意义不大），用于后面的训练  
```make_tfrecords.py```该文件用于将数据集制作成train.tfrecords和valid.tfrecords，有了tfrecords文件，训练速度大大提升  
```train_estimator.py```该文件采用estimator+tf.data.Dataset这些高级api来进行训练，代码更清晰，训练速度也很快，在我的1050Ti
显卡上，三层cnn，66x34大小的图片，一秒钟能跑3000多张图  
```train_finetune.py```该文件用于finetune，将六万多张自动生成的图片训练得到的模型，拿来finetune  
```inference.py```该文件用于inference，说来惭愧，这个事情前前后后拖了两个月才完结，搞得现在的验证码跟当时的验证码都不太一样了，
不过幸好只是尺寸变化了，resize一下还是能用的，最后的效果也还不错
### 补充说明

