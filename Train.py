from networks import Generator, Discriminator
import tensorflow as tf
import numpy as np
from utils import random_batch, random_face_batch
from PIL import Image

def Init(c_nums = 2, model_path = "./save_para//"):
    G = Generator("generator")
    z = tf.placeholder(tf.float32, [1, 100])
    train_phase = tf.placeholder(tf.bool)
    y1 = tf.placeholder(tf.float32, [1, c_nums])
    y2 = tf.placeholder(tf.float32, [1, c_nums])
    alpha = tf.placeholder(tf.float32)
    target = G(z, train_phase, y1, y2, alpha)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, model_path + ".\\model.ckpt")
    return target, sess, z, train_phase, y1, y2, alpha

def generate(z, result_path, label1, label2, alpha, target, sess, z_ph, train_phase_ph, y1_ph, y2_ph, alpha_ph):
    Y1 = np.zeros([1, 2])
    Y2 = np.zeros([1, 2])
    Y1[0, label1] = 1
    Y2[0, label2] = 1
    img = sess.run(target, feed_dict={z_ph: z, train_phase_ph: False, y1_ph: Y1, y2_ph: Y2, alpha_ph: alpha})
    Image.fromarray(np.uint8((img[0, :, :, :] + 1)*127.5)).save(result_path + "result"+str(alpha)+".jpg")

def test():
    target, sess, z, train_phase, y1, y2, alpha = Init()
    z_np = np.random.normal(0, 1, [1, 100])
    for a in range(11):
        a = a / 10
        generate(z_np, "./results//", 0, 1, a, target, sess, z, train_phase, y1, y2, alpha)

def Train(batch_size=50, z_dim=100, c_nums=2, img_h=64, img_w=64, img_c=3, lr=2e-4, beta1=0., beta2=0.9, train_itr=100000, path="./dataset//", path_train_img="./save_img//", path_save_para="./save_para//"):
    x = tf.placeholder(tf.float32, [batch_size, img_h, img_w, img_c])
    train_phase = tf.placeholder(tf.bool)
    y1 = tf.placeholder(tf.float32, [1, c_nums])
    y2 = tf.zeros([1, c_nums])
    alpha = tf.constant([1.])
    z = tf.placeholder(tf.float32, [batch_size, z_dim])
    G = Generator("generator")
    D = Discriminator("discriminator")
    fake_img = G(z, train_phase, y1, y2, alpha)
    fake_logits = D(fake_img, y1)
    real_logits = D(x, y1)
    D_loss = tf.reduce_mean(tf.maximum(0., 1 - real_logits)) + tf.reduce_mean(tf.maximum(0., 1 + fake_logits))
    G_loss = -tf.reduce_mean(fake_logits)
    D_opt = tf.train.AdamOptimizer(lr, beta1=beta1, beta2=beta2).minimize(D_loss, var_list=D.var_list())
    G_opt = tf.train.AdamOptimizer(lr, beta1=beta1, beta2=beta2).minimize(G_loss, var_list=G.var_list())
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    shape = [img_h, img_w, img_c]
    batch = 0
    Y = 0
    saver = tf.train.Saver()
    for itr in range(train_itr):
        for d in range(5):
            batch, Y = random_batch(path, batch_size, shape, c_nums)
            batch = batch / 127.5 - 1
            Z = np.random.standard_normal([batch_size, z_dim])
            sess.run(D_opt, feed_dict={x: batch, y1: Y, z: Z, train_phase: True})

        Z = np.random.standard_normal([batch_size, z_dim])
        sess.run(G_opt, feed_dict={z: Z, y1: Y, train_phase: True})
        if itr % 10 == 0:
            Dis_loss = sess.run(D_loss, feed_dict={x: batch, y1: Y, z: Z, train_phase: False})
            Gen_loss = sess.run(G_loss, feed_dict={z: Z, y1: Y, train_phase: False})
            print("Iteration: %d, D_loss: %f, G_loss: %f" % (itr, Dis_loss, Gen_loss))
            FAKE_IMG = sess.run(fake_img, feed_dict={z: Z, y1: Y, train_phase: False})
            Image.fromarray(np.uint8((FAKE_IMG[0, :, :, :] + 1)*127.5)).save(path_train_img+str(itr)+".jpg")
        if itr % 500 == 0:
            saver.save(sess, path_save_para+"model.ckpt")

if __name__ == "__main__":
    # Train()
    test()
