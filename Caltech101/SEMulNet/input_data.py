#By @Kevin Xu
#kevin28520@gmail.com
#Youtube: https://www.youtube.com/channel/UCVCSn4qQXTDAtGWpWAe4Plw
#
#The aim of this project is to use TensorFlow to process our own data.
#    - input_data.py:  read in data and generate batches
#    - model: build the model architecture
#    - training: train

# I used Ubuntu with Python 3.5, TensorFlow 1.0*, other OS should also be good.
# With current settings, 10000 traing steps needed 50 minutes on my laptop.


# data: cats vs. dogs from Kaggle
# Download link: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
# data size: ~540M

# How to run?
# 1. run the training.py once
# 2. call the run_training() in the console to train the model.

# Note: 
# it is suggested to restart your kenel to train the model multiple times 
#(in order to clear all the variables in the memory)
# Otherwise errors may occur: conv1/weights/biases already exist......


#%%

import tensorflow as tf
import numpy as np
import os

#%%

# you need to change this to your data directory# 获取文件路径和标签
## train_dir = 'C:/Users/SHAN/Desktop/ILSVRC2012/data/train/'
## label_dir = 'C:/Users/SHAN/Desktop/ILSVRC2012/data/train2.txt'

def get_files(file_dir,label_dir):
    '''
    # file_dir: 文件夹路径
    # return: 乱序后的图片和标签
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    image_list = []
    label_list = []
    len_file = len(os.listdir(file_dir))    # 判断指定文件中有几个文件
    len_label = len(open(label_dir).readlines())   # 判断train_labels_dir指定的TXT文件有多少行
    
    if len_file == len_label:
        print('num of images identify to num of labels.')
        print('The len of file is %d.'%len_file)
    # 载入数据路径并写入标签值
    txt_file = open(label_dir,'r')    # 先打开标签文件
    for file in os.listdir(file_dir):
        image_list.append(file_dir + file)
        one_content = txt_file.readline(50)
        name = one_content.split(sep=' ')
        if name[0]==file:
            name1 = name[1].split(sep='\n')
            label_list.append(int(name1[0]))
        else:
            print('Adding lebel to label_list is error!\n')
    txt_file.close()    # 关闭标签文件
    print('There are %d images\nThere are %d labels \n' %(len(image_list), len(label_list)))

##    # 串堆叠，这里不需要
##    image_list = np.hstack(cats,dogs)
##    label_list = np.hstack(cats_list,dog_list)

    # 打乱文件顺序
    temp = np.array([image_list, label_list])
    temp = temp.transpose()    # 转置
    np.random.shuffle(temp)
    
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    
    
    return image_list, label_list


#%%
# 生成相同大小的批次
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    # image, label: 要生成batch的图像和标签list
    # image_W, image_H: 图片的宽高
    # batch_size: 每个batch有多少张图片
    # capacity: 队列容量
    # return: 图像和标签的batch
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''

    # 将python.list类型转换成tf能够识别的格式
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue# 生成队列
    input_queue = tf.train.slice_input_producer([image, label])
    
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    
    ######################################
    # data argumentation should go to here
    ######################################
    # 统一图片大小
    # 视频方法
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # 我的方法
    # image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # image = tf.cast(image, tf.float32)

    # if you want to test the generated batches of images, you might want to comment(注释) the following line.
    image = tf.image.per_image_standardization(image) # 标准化数据
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, # 线程
                                                capacity = capacity)
    
    #you can also use shuffle_batch 打乱数据
#    image_batch, label_batch = tf.train.shuffle_batch([image,label],
#                                                      batch_size=BATCH_SIZE,
#                                                      num_threads=64,
#                                                      capacity=CAPACITY,
#                                                      min_after_dequeue=CAPACITY-1)
    
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch, label_batch


 
# TEST
# To test the generated batches of images
# When training the model, DO comment the following codes
# 可以用下面的代码测试获取图片是否成功，因为之前将图片转为float32了，因此这里imshow()出来的图片色彩会有点奇怪，
# 因为本来imshow()是显示uint8类型的数据（灰度值在uint8类型下是0~255，转为float32后会超出这个范围，所以色彩有点奇怪），
# 不过这不影响后面模型的训练。



#import matplotlib.pyplot as plt

#BATCH_SIZE = 2
#CAPACITY = 256
#IMG_W = 208
#IMG_H = 208
#
#train_dir = '/home/kevin/tensorflow/cats_vs_dogs/data/train/'
#
#image_list, label_list = get_files(train_dir)
#image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#
#with tf.Session() as sess:
#    i = 0
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    
#    try:
#        while not coord.should_stop() and i<1:
#            
#            img, label = sess.run([image_batch, label_batch])
#            
#            # just test one batch
#            for j in np.arange(BATCH_SIZE):
#                print('label: %d' %label[j])
#                plt.imshow(img[j,:,:,:])
#                plt.show()
#            i+=1
#            
#    except tf.errors.OutOfRangeError:
#        print('done!')
#    finally:
#        coord.request_stop()
#    coord.join(threads)


#%%





    
