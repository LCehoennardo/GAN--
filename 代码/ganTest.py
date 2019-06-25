# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 22:41:36 2019

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import os
from NewGAN import NewGAN
import matplotlib.pyplot as plt

tf.reset_default_graph()


EPOCHS = 10000
BATCH_SIZE = 10

train_dir = os.path.split(os.path.realpath(__file__))[0] + '\\train_dir'


#读取数据集，保存为npy格式
BASE_PATH = 'C:/Users/Leo/Desktop'#96*96*3
filenames = os.listdir(BASE_PATH)
CKPT_PATH = os.path.split(os.path.realpath(__file__))[0] + '\\train_dir'

imgs = []
'''
for f in filenames:
    ii = plt.imread(BASE_PATH + '/' + filenames[0])
    imgs.append(ii)
    
imgs = np.array(imgs)
'''

imgs = np.load('faces_data.npy')   #[0:20000,:,:,:]

def iter_batch(x, batch_size, shuffle=True):
    indices = np.arange(x.shape[0]) #例如，2000
    if shuffle:
        np.random.shuffle(indices)
    for i in range(0, x.shape[0], batch_size):
        yield x[indices[i:i+batch_size],:]
        

#定义计算图
gan = NewGAN()
input_size = 4*4*512
inputs_real, inputs_noise = gan.get_inputs(input_size, 96, 96, 3)

#插值参数alpha
alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1], 
        minval=0.,
        maxval=1.)

g_loss, d_loss = gan.get_loss_wgangp(inputs_real, inputs_noise, 3, alpha)
g_opt, d_opt = gan.optimizer(g_loss, d_loss)

test_noise = tf.placeholder(tf.float32, [None, input_size], name = 'test_noise')
fake_img = gan.netG(test_noise, 3, is_train=False)


init = tf.global_variables_initializer()
step = 0
saver = tf.train.Saver(max_to_keep=1)   #checkpoint saver

with tf.Session() as sess:
    sess.run(init)
    
    #先在train_dir目录中读取已训练的模型
    ckpt = tf.train.latest_checkpoint(CKPT_PATH)
    if ckpt:
        saver.restore(sess, ckpt)
        step = int(ckpt.split('\\')[-1].split('-')[-1])
        print('load checkpoint: %s' % ckpt)
    else:
        pass
    
    for epoch in range(EPOCHS):
        avg_loss = 0
        count = 0
        for x_batch in iter_batch(imgs, batch_size=BATCH_SIZE):
            if x_batch.shape[0] != BATCH_SIZE:
                break
            imgs_noise = np.random.normal(size=(x_batch.shape[0], input_size)).astype(np.float32)   #noise_img         
            
            D_loss, _ = sess.run([d_loss, d_opt], feed_dict={inputs_real: x_batch, inputs_noise: imgs_noise})
            G_loss, _ = sess.run([g_loss, g_opt], feed_dict={inputs_real: np.zeros(x_batch.shape), inputs_noise: imgs_noise})
        
            print(str(count) + '-' + str(D_loss))
        
            avg_loss += D_loss
            count += 1
            
            
        
            step = step + 1
            if step % 600 == 0:
                saver.save(sess, CKPT_PATH + '\\faces-gan.ckpt', global_step=step,write_meta_graph=False)
                print('Model at step %d saved.' % step)
            
            if step % 200 == 0:
                genImg = sess.run(fake_img, feed_dict={test_noise: np.random.normal(size=(1, input_size)).astype(np.float32)})
                plt.imshow(genImg[0]/255.)
                plt.show()
