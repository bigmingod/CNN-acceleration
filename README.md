# CNN-acceleration
*Algorithm Design and Analysis* course in the first half of 2018    
Copyright ©2018 Bigmingod pkuych and AnTuo1998. All rights reserved.   
Work of dsc are as [below](https://github.com/bigmingod/CNN-acceleration/tree/dsc).

## Ⅰ.Learning papers and codes

- [Learning Structured Sparsity in Deep Neural Network](https://arxiv.org/abs/1608.03665)    
See [caffe-scnn](https://github.com/bigmingod/CNN-acceleration/tree/dsc/caffe-scnn)

- [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding”](https://arxiv.org/abs/1510.00149)   
See [DeepCompression-caffe-master](https://github.com/bigmingod/CNN-acceleration/tree/dsc/DeepCompression-caffe-master)
and [caffe-pruned-master](https://github.com/bigmingod/CNN-acceleration/tree/dsc/caffe-pruned-master)

## Ⅱ.Sparsity Calculation
A simple [code](https://github.com/bigmingod/CNN-acceleration/tree/dsc/calculate) to count up the numbers of 0s in conv1, conv2, ip1 and ip2 of ```lenet```

## Ⅲ.Modification of ```Caffe``` Codes
1.添加新层

- 仿照卷积层，定义相应的```demo_layer.hpp```放在```include/caffe/layers，demo_layer.cpp在src/caffe/layers```目录下
- 修改```src/caffe/proto/caffe.proto```文件，添加```DemoLayer```的相关定义
- 在```src/caffe/layer_factory.cpp```里面“注册”```DemoLayer```
- 用```DemoLayer```代替```lenet```里面的```ConvolutionLayer```并编译并测试

2.添加变量函数

在```include/caffe/layer.hpp```内添加下列变量
```
vector<int> masks;
vector<Dtype> centroids;
vector<int> indices
virtual void Compute_Blob_makes(float ratio){}
```
3.定义剪枝函数

在```src/caffe/layers/demo_layer.cpp```中定义遮掩函数：
```
template <typename Dtype>
void DemoLayer<Dtype>::Compute_Blob_makes(float ratio)
```
在```src/caffe/net.cpp```的```CopyTrainedLayersFrom(const NetParameter& param)```函数中调用该函数，用于训练好的模型的初始化mask和对权值进行聚类。

4.修改向前向后传播函数

在```src/caffe/layers/demo_layer.cpp```中修改卷积层的向前向后函数：向前传播时，将权值分类进行共享，以减少相近的大量参数；向后传播时，masks掩住的权值为零不再更新，以进行剪枝。
```
template <typename Dtype>
void DemoLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom
	const vector<Blob<Dtype>*>& top)
template <typename Dtype>
void DemoLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& bottom
	const vector<Blob<Dtype>*>& top)	
```
See Details in [```demo_caffe```](https://github.com/bigmingod/CNN-acceleration/tree/dsc/demo_caffe).
