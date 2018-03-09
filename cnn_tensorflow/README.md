### 基于tensorflow框架的CNN多标签分类识别，每个验证码可以看做四分类，整体思路:
1、由于真正的样本需要标注，工作量太大，这里先用captcha自动生成六万多张图片进行训练  
2、catpcha训练得到的模型进行fine-tuning  
3、最终的模型用于inference
