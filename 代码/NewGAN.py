# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 17:37:46 2019

@author: Administrator
"""

import tensorflow as tf
import numpy as np
from tensorflow.contrib.slim.nets import resnet_v1

slim = tf.contrib.slim

nets = slim.nets

class NewGAN(object):
    #辅助函数
    def calinterpolation(self, integrated_input):
        alpha_part = integrated_input[0]
        difference_part = integrated_input[1]
        return alpha_part*difference_part

    #计算损失
    #改用wgan-gp
    def get_loss_wgangp(self, inputs_real, inputs_noise, img_depth, alpha, smooth=0.1):
        LAMBDA = 10 # Gradient penalty lambda hyperparameter

        #朴素
        '''
        g_outputs = self.netG(inputs_noise, img_depth, is_train=True)
        d_logits_real, d_outputs_real = self.netD(inputs_real)
        d_logits_fake, d_outputs_fake = self.netD(g_outputs, reuse = True)
        
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels = tf.ones_like(d_outputs_fake)*(1-smooth)))
        
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_outputs_real)*(1-smooth)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_outputs_fake)*(1-smooth)))
        
        d_loss = tf.add(d_loss_fake,d_loss_real)
        '''

        #wgan-gp
        real_data = inputs_real
        fake_data = self.netG(inputs_noise, img_depth, is_train=True)

        _, disc_real = self.netD_resnet50(real_data)
        _, disc_fake = self.netD_resnet50(fake_data, reuse=True)

        gen_cost = -tf.reduce_mean(disc_fake)
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

        differences = fake_data - real_data
        print('difference:' + str(differences))

        #对batch中的每一对fake和real进行interpolation
        integrated_tensor = (alpha, differences)
        print(integrated_tensor)
        interpolates = tf.map_fn(self.calinterpolation, elems=integrated_tensor,dtype=tf.float32)

        interpolates = real_data + interpolates
        gradients = tf.gradients(self.netD_resnet50(interpolates,reuse=True), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        disc_cost += LAMBDA*gradient_penalty

        return gen_cost, disc_cost

    #输入
    def get_inputs(self, noise_dim, img_h, img_w, img_depth):
        inputs_real = tf.placeholder(tf.float32, [None, img_h, img_w, img_depth], name = 'inputs_real')
        inputs_noise = tf.placeholder(tf.float32, [None, noise_dim], name = 'inputs_noise')
        return inputs_real, inputs_noise

    #返回一幅图像，输入大小是input_size
    def netG(self, noise_img, output_dim, is_train = True, alpha = 0.1):
        with tf.variable_scope('netG', reuse = (not is_train)):
            #layer1
            lay1 = tf.layers.dense(noise_img, 4*4*512)
            lay1 = tf.reshape(lay1, [-1, 4, 4, 512])
            lay1 = tf.layers.batch_normalization(lay1, training=is_train)
            lay1 = tf.maximum(alpha * lay1, lay1)
            lay1 = tf.layers.dropout(lay1, rate=0.2)
            #4*4*512

            lay2 = tf.layers.conv2d_transpose(lay1, 256, 4, strides=1, padding='VALID')
            lay2 = tf.layers.batch_normalization(lay2, training=is_train)
            lay2 = tf.maximum(alpha * lay2, lay2)
            lay2 = tf.layers.dropout(lay2, rate=0.2)
            #7*7*256

            lay3 = tf.layers.conv2d_transpose(lay2, 256, 4, strides=1, padding='VALID')
            lay3 = tf.layers.batch_normalization(lay3, training=is_train)
            lay3 = tf.maximum(alpha * lay3, lay3)
            lay3 = tf.layers.dropout(lay3, rate=0.2)
            #10*10*256

            lay4 = tf.layers.conv2d_transpose(lay3, 128, 3, strides=2, padding='SAME')
            lay4 = tf.layers.batch_normalization(lay4, training=is_train)
            lay4 = tf.maximum(alpha * lay4, lay4)
            lay4 = tf.layers.dropout(lay4, rate=0.2)
            #20*20*128

            lay5 = tf.layers.conv2d_transpose(lay4, 128, 3, strides=1, padding='VALID')
            lay5 = tf.layers.batch_normalization(lay5, training=is_train)
            lay5 = tf.maximum(alpha * lay5, lay5)
            lay5 = tf.layers.dropout(lay5, rate=0.2)
            #22*22*128

            lay6 = tf.layers.conv2d_transpose(lay5, 64, 3, strides = 2, padding='SAME')
            lay6 = tf.layers.batch_normalization(lay6, training=is_train)
            lay6 = tf.maximum(alpha * lay6, lay6)
            lay6 = tf.layers.dropout(lay6, rate=0.2)
            #44*44*64

            lay7 = tf.layers.conv2d_transpose(lay6, 32, 5, strides=1, padding='VALID')
            lay7 = tf.layers.batch_normalization(lay7, training=is_train)
            lay7 = tf.maximum(alpha * lay7, lay7)
            lay7 = tf.layers.dropout(lay7, rate=0.2)
            #48*48*32

            lay8 = tf.layers.conv2d_transpose(lay7, 16, 3, strides = 2, padding='SAME')
            lay8 = tf.layers.batch_normalization(lay8, training=is_train)
            lay8 = tf.maximum(alpha * lay8, lay8)
            lay8 = tf.layers.dropout(lay8, rate=0.2)
            #96*96*16

            #output_dim是3，RGB图像
            #第三个参数是：
            logits = tf.layers.conv2d_transpose(lay8, output_dim, 3, strides=1, padding='SAME')

            #对输出进行变换，值在0-255之间
            outputs = (tf.sigmoid(logits) + 1) * 127.5

            return outputs


    '''
    def netG(self, noise_img, output_dim, is_train = True, alpha = 0.1):
        with tf.variable_scope('netG', reuse = (not is_train)):
            #layer1
            lay1 = tf.layers.dense(noise_img, 4*4*512)
            lay1 = tf.reshape(lay1, [-1, 4, 4, 512])
            lay1 = tf.layers.batch_normalization(lay1, training=is_train)
            lay1 = tf.maximum(alpha * lay1, lay1)
            #dropout
            lay1 = tf.layers.dropout(lay1, rate=0.2)
            
            #layer2-8*8*256
            lay2 = tf.layers.conv2d_transpose(lay1, 256, 3, strides=2,padding='SAME')
            lay2 = tf.layers.batch_normalization(lay2, training=is_train)
            lay2 = tf.maximum(alpha * lay2, lay2)
            lay2 = tf.layers.dropout(lay2, rate=0.2)
            
            #layer3-16*16*256
            lay3 = tf.layers.conv2d_transpose(lay2, 256, 3, strides = 2,padding='SAME')
            lay3 = tf.layers.batch_normalization(lay3, training=is_train)
            lay3 = tf.maximum(alpha * lay3, lay3)
            lay3 = tf.layers.dropout(lay3, rate=0.2)
            
            #layer4-32*32*128
            lay4 = tf.layers.conv2d_transpose(lay3, 128, 3, strides = 2, padding='SAME')
            lay4 = tf.layers.batch_normalization(lay4, training=is_train)
            lay4 = tf.maximum(alpha * lay4, lay4)
            lay4 = tf.layers.dropout(lay4, rate=0.2)
            
            #layer5-96*96*64
            lay5 = tf.layers.conv2d_transpose(lay4, 64, 3, strides = 3, padding='SAME')
            lay5 = tf.layers.batch_normalization(lay5, training=is_train)
            lay5 = tf.maximum(alpha * lay5, lay5)
            lay5 = tf.layers.dropout(lay5, rate=0.2)
            
            #layer5
            #output_dim是3，RGB图像
            #第三个参数是：
            logits = tf.layers.conv2d_transpose(lay5, output_dim, 3, strides=1, padding='SAME')
            
            #对输出进行变换，值在0-255之间
            outputs = (tf.sigmoid(logits) + 1) * 127.5
            
            return outputs
    '''

    #判别器
    def netD(self, inputs_img, reuse=False, alpha=0.01):
        with tf.variable_scope('netD', reuse=reuse):
            lay1 = tf.layers.conv2d(inputs_img, 64, 3, strides=2, padding='SAME')
            lay1 = tf.maximum(alpha * lay1, lay1)
            lay1 = tf.layers.dropout(lay1, rate=0.2)

            lay2 = tf.layers.conv2d(lay1, 128, 3, strides=2, padding='SAME')
            lay2 = tf.layers.batch_normalization(lay2, training=True)
            lay2 = tf.maximum(alpha * lay2, lay2)
            lay2 = tf.layers.dropout(lay2, rate=0.2)

            lay3 = tf.layers.conv2d(lay2, 256, 3, strides=2, padding='SAME')
            lay3 = tf.layers.batch_normalization(lay3, training=True)
            lay3 = tf.maximum(alpha * lay3, lay3)
            lay3 = tf.layers.dropout(lay3,rate=0.2)

            lay4 = tf.layers.conv2d(lay3, 512, 3, strides=2, padding='SAME')
            lay3 = tf.layers.batch_normalization(lay3, training=True)
            lay4 = tf.maximum(alpha * lay4, lay4)
            lay4 = tf.layers.dropout(lay4,rate=0.2)

            #拉平
            flatten = tf.layers.Flatten()(lay4)
            logits = tf.layers.dense(flatten, 1)

            outputs = tf.sigmoid(logits)    #压缩到(0,1)区间

            return logits, outputs

    def netD_resnet50(self, inputs_img, reuse=False):
        with tf.variable_scope('netD_resnet50', reuse=reuse):
            #logits是一个(?,1,1,2048)的tensor
            logits, end_points = nets.resnet_v2.resnet_v2_50(inputs_img, is_training=True)
            logits = tf.layers.dense(logits, 1)
            outputs = tf.sigmoid(logits)

            return logits, outputs


    def optimizer(self, g_loss, d_loss, beta1=0.4, learning_rate = 0.001):
        train_vars = tf.trainable_variables()

        g_vars = [var for var in train_vars if var.name.startswith('netG')]
        d_vars = [var for var in train_vars if var.name.startswith('netD')]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)
            d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)

        return g_opt, d_opt

'''
tf.reset_default_graph()    
imgs_noise = np.random.normal(size=(1,4*4*512)).astype(np.float32)
imgs_tensor = tf.convert_to_tensor(imgs_noise)
gan = NewGAN()
fake_img = gan.netG(imgs_tensor, 3)
'''


