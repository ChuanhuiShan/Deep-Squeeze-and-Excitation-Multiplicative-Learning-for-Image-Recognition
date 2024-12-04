#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 09:06:42 2018
@author: xy
"""
import tensorflow as tf 
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import os
import pickle
import time

iteration_steps = 500000
batch_size = 32

is_train = True

all_step = []
all_loss = []
all_train_accuracy = []

#下载minist数据，创建mnist_data文件夹，one_hot编码
def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f,encoding='latin1')
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32,32).transpose(0,2,3,1).astype(np.float32)
    X = X / 128.0 - 1
    Y = np.array(Y)
    one_hot_labels = []
    # print(len(Y))
    for num in Y:
      one_hot = [0.] * 10
      one_hot[num % 10] = 1.
      one_hot_labels.append(one_hot)
    Y1 = np.array(one_hot_labels).astype(np.float32)
    
    # Y = np.array([x[0] for x in data_set["y"]])
    # one_hot_labels = []
    # for num in Y:
        # one_hot = [0.] * 10
        # one_hot[num % 10] = 1.
        # one_hot_labels.append(one_hot)
    # Y = np.array(one_hot_labels).astype(np.float32)    
    return X, Y1

def load(file):
    data_set = loadmat(file)
    samples = np.transpose(data_set["X"], (3, 0, 1, 2)).astype(np.float32)
    # print(samples.shape)
    labels = np.array([x[0] for x in data_set["y"]])
    one_hot_labels = []
    for num in labels:
        one_hot = [0.] * 10
        one_hot[num % 10] = 1.
        one_hot_labels.append(one_hot)
    labels = np.array(one_hot_labels).astype(np.float32)
    # samples = np.add.reduce(samples, keepdims=True, axis=3) / 3.0
    samples = samples / 128.0 - 1
    return samples, labels

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)
  Xtr = np.concatenate(xs)#使变成行向量
  Ytr = np.concatenate(ys)
  Ytr.astype(np.float32)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  Yte.astype(np.float32)
  return Xtr, Ytr, Xte, Yte

def data_iterator(samples, labels, batch_size, iteration_steps=None):
  if len(samples) != len(labels):
    raise Exception('Length of samples and labels must equal')
  if len(samples) < batch_size:
    raise Exception('Length of samples must be smaller than batch size.')
  start = 0
  step = 0
  if iteration_steps is None:
    while start < len(samples):
      end = start + batch_size
      if end < len(samples):
        yield step, samples[start:end], labels[start:end]
        step += 1
      start = end
  else:
    while step < iteration_steps:
      start = (step * batch_size) % (len(labels) - batch_size)
      yield step, samples[start:start + batch_size], labels[start:start + batch_size]
      step += 1

root = './cifar-10-batches-py'
train_samples, train_labels, test_samples, test_labels =  load_CIFAR10(root)

print("The shape of train samples:", train_samples.shape)
print("The shape of train labels:", train_labels.shape)
print("The shape of test samples:", test_samples.shape)
print("The shape of test labels:", test_labels.shape)


# 开辟结构用来给自定义函数使用
x = tf.placeholder(tf.float32, [None, 32, 32, 3])                        
y_real = tf.placeholder(tf.float32, shape=[None, 10])

#初始化权重
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.05)#正态分布
  return tf.Variable(initial)
#初始化偏置
def bias_variable(shape):
  initial = tf.constant(0.05, shape=shape)#偏置初始化为0.1
  return tf.Variable(initial)
#构建卷积层
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')#卷积步长为1,不足补0
  
def conv1d(x, W):
  return tf.nn.conv1d(x, W, 1, "VALID")#卷积步长为1,不足补0
  
#构建池化层
def max_pool(x):
    #大小2*2,步长为2,不足补0
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
  
def matrix_product_1w(xx, WW):
  xx = tf.split(xx, 3, 3, name='split') # (?,4,4,1)
  WW = tf.split(WW, 3, 2, name='split') # (4,3,1,1)
  x2 = 0
  for i in range(3):
    xx[i] = tf.squeeze(xx[i], 3) # x[0].shape = [-1,4,4]
    WW[i] = tf.squeeze(WW[i], 3) # W1.shape = (4,3,1)
    # tf.einsum('ibh,hnd->ibnd', left, right)
    x1 = tf.einsum('ibh,hnd->ibnd', xx[i], WW[i])  # x1.shape = (?,4,3,1)
    # print('x1_size:', x1.get_shape().as_list())
    x2 = tf.add(x1, x2)
  return x2

#第一层
x_image = tf.reshape(x, [-1,32,32,3]) 
W_conv1 = weight_variable([5, 5, 3, 128])      
b_conv1 = bias_variable([128])       
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)#卷积层
h_pool1 = max_pool(h_conv1)#池化层

#第二层:Res第一层
W_conv2 = weight_variable([3, 3, 128, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)      

#第三层:Res第二层
W_conv3 = weight_variable([3, 3, 64, 64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3) 

#第四层:Res第三层
W_conv4 = weight_variable([3, 3, 64, 128])
b_conv4 = bias_variable([128])
h_conv4 = conv2d(h_conv3, W_conv4) + b_conv4     

#######################################################
glob_pool1 = tf.reduce_mean(h_pool1, [1, 2], name='global_pool', keep_dims=True) # Global pooling
glob_pool_flat1 = tf.reshape(glob_pool1, [-1, 128])   

Se_W_fc1 = weight_variable([128, 32])
Se_fc1 = tf.nn.relu(tf.matmul(glob_pool_flat1, Se_W_fc1))

Se_W_fc2 = weight_variable([32, 128])
Se_fc2 = tf.sigmoid(tf.matmul(Se_fc1, Se_W_fc2))

Se_fc2_tensor1 = tf.reshape(Se_fc2, [-1, 1, 1, 128])  
##################################################################

h_conv5 = tf.multiply(h_conv4, Se_fc2_tensor1)

h_pool2 = max_pool(h_conv5)


# #第一层
# x_image = tf.reshape(x, [-1,32,32,3]) 
# W_conv1 = weight_variable([5, 5, 3, 32])      
# b_conv1 = bias_variable([32])       
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)#卷积层
# #########################################
# W_se1 = weight_variable([1,1, 32])
# h_se1 = tf.multiply(h_conv1, tf.sigmoid(tf.layers.batch_normalization(W_se1, training=is_train)))
# # h_se1 = tf.multiply(h_conv1, tf.sigmoid(batch_norm(W_se1, decay=0.9,updates_collections=None,is_training=is_train)))

# #########################################
# h_pool1 = max_pool(h_se1)#池化层

# #第二层
# W_conv2 = weight_variable([5, 5, 32, 64])
# b_conv2 = bias_variable([64])
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)      
# #########################################
# W_se2 = weight_variable([1,1, 64])
# h_se2 = tf.multiply(h_conv2, tf.sigmoid(tf.layers.batch_normalization(W_se2, training=is_train)))
# # h_se2 = tf.multiply(h_conv2, tf.sigmoid(batch_norm(W_se2, decay=0.9,updates_collections=None,is_training=is_train)))

# #########################################
# h_pool2 = max_pool(h_se2)

[p, wi, hi, q] = h_pool2.get_shape().as_list()
#密集连接层
W_fc1 = weight_variable([wi * hi * q, 1024])
b_fc1 = bias_variable([1024])
h_pool1_flat = tf.reshape(h_pool2, [-1, wi * hi * q])              
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1))    
#dropout
keep_prob = tf.placeholder("float") 
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)                 
#输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_predict=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) 
print(y_predict.get_shape().as_list())  
#模型训练评估
cross_entropy = -tf.reduce_sum(y_real*tf.log(y_predict))    
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy) 
logs_loss=-tf.reduce_sum(tf.reduce_mean(y_real*tf.log(y_predict)))   
correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_real,1))    
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))                
sess=tf.InteractiveSession()                          
sess.run(tf.global_variables_initializer())
#for i in range(50001):
for step, samples, labels in data_iterator(train_samples, train_labels, batch_size, iteration_steps):
  # batch = mnist.train.next_batch(50)
  # batch = next_batch([train_samples, train_labels])
  if step%100 == 0:#训练100次
    train_loss = logs_loss.eval(feed_dict={x:samples, y_real: labels, keep_prob: 1.0})
    train_accuracy = accuracy.eval(feed_dict={x:samples, y_real: labels, keep_prob: 1.0})
    print('step %d,train_loss %g, training accuracy %g'%(step, train_loss, train_accuracy))
    all_step.append(step)
    all_loss.append(train_loss)
    all_train_accuracy.append(train_accuracy)
    train_step.run(feed_dict={x: samples, y_real: labels, keep_prob: 0.5})
  if step%100000 == 0:
    test_accuracy=accuracy.eval(feed_dict={x: test_samples, y_real: test_labels, keep_prob: 1.0})
    print("test accuracy",test_accuracy)

np.save('./all_step.npy',all_step)
np.save('./all_loss.npy',all_loss)
np.save('./all_train_accuracy.npy',all_train_accuracy)

is_train = False 
test_accuracy=accuracy.eval(feed_dict={x: test_samples, y_real: test_labels, keep_prob: 1.0})
print("test accuracy",test_accuracy)