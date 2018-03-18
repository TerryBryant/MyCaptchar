### 采用传统图像处理的方法进行验证码识别，基本步骤如下
1、图像二值化  
2、验证码字符分割  
3、验证码字符逐个识别，给出最终结果  
下面一步一步按照流程开始进行识别：  
## 1、图像二值化
这里仔细查看了原始验证码的RGB通道像素值，发现在该通道无法直接用门限进行二值化，否则会有背景干扰。这里考虑将图像转到HSV通道，
发现刚好有门限可以将字符和背景区分开，如下图所示
## 2、验证码字符分割
经过二值化处理后，还需将字符拆分开，逐个字符进行模板匹配。但是这里发现字符存在互相粘在一起的情况，这对字符分割产生了较大困难。
在网上查了下，发现有关拆分字符都可以做专门的算法研究了。为了省事，这里采用一种相对简便的方法，虽然不能完全正确的将字符分割开，
但多数情况下效果都还不错。具体步骤如下：  
1、