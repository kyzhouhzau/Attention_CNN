# Relation Classification via Multi-Level Attention CNNs
## 这份代码用于实现以上这篇文章。
## 数据集 ：SemEval-2010
## 框架： Tensorflow==1.9

## 实践中发现当输入层使用了input——attention后效果反而变差，检查代码并没发现明显错误。

### 使用方法：

1. 训练模型：
	bash run.sh

2. 测试结果：
	bash eval.sh
 
参考：https://github.com/FrankWork/acnn
