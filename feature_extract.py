import mxnet as mx
import numpy as np
import csv

import argparse

# extract features from pretrained model inception

# note: parameter --num-epochs <number> is needed

#parse args
parser = argparse.ArgumentParser(description='train an image classifer')
parser.add_argument('--gpus', type=str, default='0',  
                    help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--batchsize', type=int, default=64,  
                    help='the batch size')
parser.add_argument('--learningrate', type=float, default=.01,  
                    help='the initial learning rate')
parser.add_argument('--modelprefix', type=str,  
                    help='the prefix of the model to ../save')
parser.add_argument('--numepochs', required=True, type=int, default=50,  
                    help='the number of training epochs')                   
parser.add_argument('--loadepoch', type=int,  
                    help="load the model on an epoch using the model-prefix")
args = parser.parse_args()

#准备一些变量
batch_size = args.batchsize
epoch = args.numepochs
prefix = "../model/Inception-7"
data_shape = (3, 224, 224)

# 创建训练集iter
train_iter = mx.io.ImageRecordIter(
    path_imgrec="../list/rumor_vs_nonrumor_train.rec",
    data_shape=data_shape,       #期望的数据形状，注意: 即使图片不是这个尺寸，也可以在此被自动转换
    batch_size=batch_size      #每次传入1000条数据
)

# 创建测试集iter 
val_iter = mx.io.ImageRecordIter(
    path_imgrec="../list/rumor_vs_nonrumor_test.rec",
    data_shape=data_shape,
    batch_size=batch_size     #必须与上面的batch_size相等，否则不能对应
)

model_load = mx.model.FeedForward.load(prefix, epoch)

#提取特征
internals = model_load.symbol.get_internal()
fea_symbol = internals["flatten_output"]     #输出特征的层
feature_extractor = mc.model.FeedForward(ctx=mx.cpu(), symbol = fea_symbol, numpy_batch_size = batch_size, arg_params=model_load.arg_params, aux_params = model_load.aux_params, allow_extra_params=True)

#[train_feature, traindata, trainlabel]= feature_extractor.predict(train_iter, return_data=True)
[val_feature, valdata, vallabel]= feature_extractor.predict(val_iter, return_data=True)

#with open("../features/test_image_feature.csv", "w") as f:     #保存train图像的特征，后续加上 tweetId
with open("../features/test_image_feature.csv", "w") as f:     #保存test图像的特征，后续加上 tweetId 
    writer = csv.writer(f)
    writer.writerows(val_feature)
#    writer.writerows(train_feature)
