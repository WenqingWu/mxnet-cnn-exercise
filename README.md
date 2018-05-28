# mxnet-cnn-exercise

## file tree：

code/
├── feature_extract.py
├── gluon_cnn.py
├── im2rec.py
├── listgen.py
└── preprocess.py

其中，listgen.py,  im2rec.py用于图像数据的预处理

### listgen.py:

分别根据根据trainData/ ，testData/目录的json文件，生成对应.lst文件

### im2rec.py:
Im2rec是mxnet提供的小工具，用在这里是为了根据上述lst文件生成.rec文件

###gluon_cnn.py:

用mxnet搭建CNN网络,根据数据集训练模型，并将训练好的模型保存到WeiboRumorSet/save/目录下

### feature_extract.py:

使用预训练模型进行特征提取，并将特征保存为csv文件
