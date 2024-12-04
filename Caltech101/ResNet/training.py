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

import os
import numpy as np
import tensorflow as tf
import input_data
import model

#%%

N_CLASSES = 101
IMG_W = 224  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 224
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 16
CAPACITY = 2000
MAX_STEP = 5000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001 # with current parameters, it is suggested to use learning rate<0.0001
# size_val = 1508

all_step = []
all_tra_loss = []
all_tra_acc = []


def net_evaluation(input_images, input_labels, is_training, sess, cost, accuracy, test_images, test_labels):
    # input_images = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
    # 图片标签
    # input_labels = tf.placeholder(dtype=tf.int32, shape=[None, NUM_CLASSES])
    # 训练还是测试？测试的时候弃权参数会设置为1.0
    # is_training = tf.placeholder(dtype=tf.bool)

    #val_nums = 0
    #for record in tf.python_io.tf_record_iterator(val_record_file):
    #    val_nums += 1
    val_nums = get_example_nums(val_record_file)
    val_max_steps = int(val_nums / batch_size)  # 测试集需要迭代的次数

    val_losses = []
    val_accs = []
    # 在所有测试图片上迭代，获得平均loss，平均准确度
    for _ in range(val_max_steps):
        test_imgs, test_labs = sess.run([test_images, test_labels])
        val_loss, val_acc = sess.run([cost, accuracy],
                                     feed_dict={input_images: test_imgs, input_labels: test_labs,
                                                is_training: False
                                                })
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    mean_loss = np.array(val_losses, dtype='float32').mean()
    mean_acc = np.array(val_accs, dtype='float32').mean()
    return mean_loss, mean_acc


#%%
def run_training():
    
    # you need to change the directories to yours.
    train_dir = './data/train/'
    train_label_dir = './data/train.txt'
    logs_train_dir = './data/logs/train/'
    
    val_dir = './data/val/'
    val_label_dir = './data/val.txt'
    logs_val_dir = './data/logs/val/'
    
    train, train_label = input_data.get_files(train_dir,train_label_dir)
    val, val_label = input_data.get_files(val_dir,val_label_dir)
    
    train_batch, train_label_batch = input_data.get_batch(train,
                                                          train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          TRAIN_BATCH_SIZE, 
                                                          CAPACITY)
    val_batch, val_label_batch = input_data.get_batch(val,
                                                          val_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          VAL_BATCH_SIZE, 
                                                          CAPACITY)
    
    logits = model.inference(train_batch, TRAIN_BATCH_SIZE, N_CLASSES)
    loss = model.losses(logits, train_label_batch)        
    train_op = model.trainning(loss, learning_rate)
    acc = model.evaluation(logits, train_label_batch)
    
    x_train = tf.placeholder(tf.float32, shape=[TRAIN_BATCH_SIZE, IMG_W, IMG_H, 3])
    y_train_ = tf.placeholder(tf.int16, shape=[TRAIN_BATCH_SIZE])

    x_val = tf.placeholder(tf.float32, shape=[VAL_BATCH_SIZE, IMG_W, IMG_H, 3])
    y_val_ = tf.placeholder(tf.int16, shape=[VAL_BATCH_SIZE])
	
    #############################################################################
    #  last test
    # val_all, val_label_all = input_data.get_batch(val,
                                                      # val_label,
                                                      # IMG_W,
                                                      # IMG_H,
                                                      # size_val, 
                                                      # CAPACITY)
    # x_val_all = tf.placeholder(tf.float32, shape=[size_val, IMG_W, IMG_H, 3])
    # y_val_all = tf.placeholder(tf.int16, shape=[size_val])
    
    #############################################################################	
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        ## 判断ckpt是否为空，若不为空，才进行模型的加载，否则从头开始训练 
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)  
        if ckpt and ckpt.model_checkpoint_path:
            print('Retrain exist!')
            saver.restore(sess,ckpt.model_checkpoint_path)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)
        
        summary_op = tf.summary.merge_all()        
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)
    
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                        break
                
                tra_images,tra_labels = sess.run([train_batch, train_label_batch])
                _, tra_loss, tra_acc = sess.run([train_op, loss, acc],
                                                feed_dict={x_train:tra_images, y_train_:tra_labels})
                if step % 50 == 0:
                    print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
                    all_step.append(step)
                    all_tra_loss.append(tra_loss)
                    all_tra_acc.append(tra_acc*100.0)
                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str, step)
                    
                if step % 500 == 0 or (step + 1) == MAX_STEP:
                    val_images, val_labels = sess.run([val_batch, val_label_batch])
                    val_loss, val_acc = sess.run([loss, acc], 
                                                 feed_dict={x_val:val_images, y_val_:val_labels})
                    print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc*100.0))
                    # summary_str = sess.run(summary_op)
                    # val_writer.add_summary(summary_str, step)  
                                    
                if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                if (step + 1) == MAX_STEP:
                    val_nums = len(os.listdir(val_dir))
                    val_max_steps = int(val_nums / VAL_BATCH_SIZE)
                    val_losses = []
                    val_accs = []
                    for _ in range(val_max_steps):
                        val_images, val_labels = sess.run([val_batch, val_label_batch])
                        val_loss, val_acc = sess.run([loss, acc], 
                                                 feed_dict={x_val:val_images, y_val_:val_labels})
                        val_losses.append(val_loss)
                        val_accs.append(val_acc)
                    mean_loss = np.array(val_losses, dtype='float32').mean()
                    mean_acc = np.array(val_accs, dtype='float32').mean()
                    print('**Last Step %d, all_val mean loss = %.2f, all_val mean accuracy = %.2f%%  **' %(step, mean_loss, mean_acc*100.0))
            np.save('./all_step.npy',all_step)
            np.save('./all_tra_loss.npy',all_tra_loss)
            np.save('./all_tra_acc.npy',all_tra_acc)                    
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()           
            coord.join(threads)

#%% Evaluate one image
# when training, comment the following codes.


#from PIL import Image
#import matplotlib.pyplot as plt
#
#def get_one_image(train):
#    '''Randomly pick one image from training data
#    Return: ndarray
#    '''
#    n = len(train)
#    ind = np.random.randint(0, n)
#    img_dir = train[ind]
#
#    image = Image.open(img_dir)
#    plt.imshow(image)
#    image = image.resize([208, 208])
#    image = np.array(image)
#    return image
#
#def evaluate_one_image():
#    '''Test one image against the saved models and parameters
#    '''
#    
#    # you need to change the directories to yours.
#    train_dir = '/home/kevin/tensorflow/cats_vs_dogs/data/train/'
#    train, train_label = input_data.get_files(train_dir)
#    image_array = get_one_image(train)
#    
#    with tf.Graph().as_default():
#        BATCH_SIZE = 1
#        N_CLASSES = 2
#        
#        image = tf.cast(image_array, tf.float32)
#        image = tf.image.per_image_standardization(image)
#        image = tf.reshape(image, [1, 208, 208, 3])
#        logit = model.inference(image, BATCH_SIZE, N_CLASSES)
#        
#        logit = tf.nn.softmax(logit)
#        
#        x = tf.placeholder(tf.float32, shape=[208, 208, 3])
#        
#        # you need to change the directories to yours.
#        logs_train_dir = '/home/kevin/tensorflow/cats_vs_dogs/logs/train/' 
#                       
#        saver = tf.train.Saver()
#        
#        with tf.Session() as sess:
#            
#            print("Reading checkpoints...")
#            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
#            if ckpt and ckpt.model_checkpoint_path:
#                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#                saver.restore(sess, ckpt.model_checkpoint_path)
#                print('Loading success, global_step is %s' % global_step)
#            else:
#                print('No checkpoint file found')
#            
#            prediction = sess.run(logit, feed_dict={x: image_array})
#            max_index = np.argmax(prediction)
#            if max_index==0:
#                print('This is a cat with possibility %.6f' %prediction[:, 0])
#            else:
#                print('This is a dog with possibility %.6f' %prediction[:, 1])


#%%

run_training()



