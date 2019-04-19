from networks_32 import Generator, Discriminator
from ops import Hinge_loss
import tensorflow as tf
import numpy as np
from utils import random_batch, read_cifar
from PIL import Image
import time
import scipy.io as sio

NUMS_CLASS = 10
TRAIN_ITR = 100000
BATCH_SIZE = 64
Z_DIM = 128

def Train():
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    train_phase = tf.placeholder(tf.bool)
    z = tf.placeholder(tf.float32, [None, 128])
    y = tf.placeholder(tf.int32, [None])
    G = Generator("generator")
    D = Discriminator("discriminator")
    fake_img = G(z, train_phase, y, NUMS_CLASS)
    fake_logits = D(fake_img, y, NUMS_CLASS, None)
    real_logits = D(x, y, NUMS_CLASS, "NO_OPS")
    D_loss, G_loss = Hinge_loss(real_logits, fake_logits)
    D_opt = tf.train.AdamOptimizer(2e-4, beta1=0., beta2=0.9).minimize(D_loss, var_list=D.var_list())
    G_opt = tf.train.AdamOptimizer(2e-4, beta1=0., beta2=0.9).minimize(G_loss, var_list=G.var_list())
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # saver.restore(sess, path_save_para+".\\model.ckpt")
    data = np.concatenate((sio.loadmat("./dataset/data_batch_1.mat")["data"], sio.loadmat("./dataset/data_batch_2.mat")["data"],
           sio.loadmat("./dataset/data_batch_3.mat")["data"], sio.loadmat("./dataset/data_batch_4.mat")["data"],
           sio.loadmat("./dataset/data_batch_5.mat")["data"]), axis=0)
    data = np.reshape(data, [50000, 3, 32, 32])
    data = np.transpose(data, axes=[0, 2, 3, 1])
    labels = np.concatenate((sio.loadmat("./dataset/data_batch_1.mat")["labels"], sio.loadmat("./dataset/data_batch_2.mat")["labels"],
           sio.loadmat("./dataset/data_batch_3.mat")["labels"], sio.loadmat("./dataset/data_batch_4.mat")["labels"],
           sio.loadmat("./dataset/data_batch_5.mat")["labels"]), axis=0)[:, 0]
    for itr in range(TRAIN_ITR):
        readtime = 0
        updatetime = 0
        for d in range(5):
            s_read = time.time()
            batch, Y = read_cifar(data, labels, BATCH_SIZE)
            e_read = time.time()
            readtime += e_read - s_read
            batch = batch / 127.5 - 1
            Z = np.random.standard_normal([BATCH_SIZE, Z_DIM])
            s_up = time.time()
            sess.run(D_opt, feed_dict={x: batch, z: Z, train_phase: True, y: Y})
            e_up = time.time()
            updatetime += e_up - s_up

        Z = np.random.standard_normal([BATCH_SIZE, Z_DIM])
        s = time.time()
        sess.run(G_opt, feed_dict={z: Z, train_phase: True, y: Y})
        e = time.time()
        one_itr_time = e - s + updatetime + readtime
        if itr % 100 == 0:
            Dis_loss = sess.run(D_loss, feed_dict={x: batch, z: Z, train_phase: False, y: Y})
            Gen_loss = sess.run(G_loss, feed_dict={z: Z, train_phase: False, y: Y})
            print("Iteration: %d, D_loss: %f, G_loss: %f, Read_time: %f, Updata_time: %f, One_itr_time: %f" % (itr, Dis_loss, Gen_loss, readtime, updatetime, one_itr_time))
            FAKE_IMG = sess.run(fake_img, feed_dict={z: Z, train_phase: False, y: Y})
            Image.fromarray(np.uint8((FAKE_IMG[0, :, :, :] + 1)*127.5)).save("./save_img/"+str(itr) + "_" + str(int(Y[0])) + ".jpg")
        if itr % 500 == 0:
            saver.save(sess, "./save_para/model.ckpt")

if __name__ == "__main__":
    Train()