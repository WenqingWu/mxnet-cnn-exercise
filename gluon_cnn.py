import mxnet as mx
import numpy as np

import argparse
import logging

logging.getLogger().setLevel(logging.DEBUG)

# 1. design model
# 2. train model (& save trained model)

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


def cnn_net():

    data = mx.symbol.Variable('data')
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    pool1 = mx.symbol.Pooling(data=relu1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(4,4), num_filter=50)
    relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, pool_type="max",
                              kernel=(2,2), stride=(2,2))

    # first fullc
    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    relu4 = mx.symbol.Activation(data=fc1, act_type="relu")

    # second fullc
    fc2 = mx.symbol.FullyConnected(data=relu4, num_hidden=2)
    # loss
    softmax = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return softmax


batch_size = args.batchsize
data_shape = (3, 224, 224)

model = mx.model.FeedForward(
    ctx = mx.cpu(0),            #使用gpu，若无gpu写mx.cpu()
    symbol = cnn_net(),       #使用cnn
    num_epoch = args.num_epochs,             #运行epoch函数，全样本遍历一次是一个epoch
    learning_rate = 0.01       #学习率，可以调整
)

# ImageRecordIter是一种特殊的DataIter，可以直接接受上文所说的数据集
# 转化成4维NDArray。读取rec文件生成一个ImageRecordIter：
# 创建训练集iter
train_iter = mx.io.ImageRecordIter(
    path_imgrec="../list/rumor_vs_nonrumor_train.rec",
    data_shape=data_shape,       #期望的数据形状，注意: 即使图片不是这个尺寸，也可以在此被自动转换
    batch_size=batch_size,      #每次传入1000条数据
)

# 创建测试集iter 
val_iter = mx.io.ImageRecordIter(
    path_imgrec="../list/rumor_vs_nonrumor_test.rec",
    data_shape=data_shape,
    batch_size=batch_size,     #必须与上面的batch_size相等，否则不能对应
)

#train 训练过程
model.fit(
    X = train_iter,           #训练集
    eval_data = val_iter,     #验证集
    batch_end_callback = mx.callback.Speedometer(batch_size, 200)      #监视训练状态，每200个iteration输出一次
)

# features = model.get_outputs()
# internals = model.symbol.get_internal()
# fea_symbol = internals["flatten_output"]     #choose feature layer

prefix = '../save/cnn_model'  
iteration = args.numepochs  
model.save(prefix, iteration)  

print ('score on test set:', model.score(val_iter))