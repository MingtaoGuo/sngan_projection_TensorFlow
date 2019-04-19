# sngan_projection_TensorFlow
Simply implementing the paper: cGANs with Projection Discriminator

## Introduction
SNGAN with projection discriminator implemented by TensorFlow. The paper [cGANs with Projection Discriminator](https://arxiv.org/pdf/1802.05637v1.pdf)

![](https://github.com/MingtaoGuo/sngan_projection_TensorFlow/blob/master/IMAGES/projection.jpg)
## Results
### More results are under training ......
### Generate cifar-10
![](https://github.com/MingtaoGuo/sngan_projection_TensorFlow/blob/master/IMAGES/cifar.jpg)

As shown in below is trained about 10000 iterations with batch size of 64.

![](https://github.com/MingtaoGuo/sngan_projection_TensorFlow/blob/master/IMAGES/generate.jpg)
#### Consecutive category morphing with fixed z:
|cat2human|cat2human|zi2zi|zi2zi|
|-|-|-|-|
|![](https://github.com/MingtaoGuo/sngan_projection_TensorFlow/blob/master/IMAGES/1.gif)|![](https://github.com/MingtaoGuo/sngan_projection_TensorFlow/blob/master/IMAGES/2.gif)|![](https://github.com/MingtaoGuo/sngan_projection_TensorFlow/blob/master/IMAGES/character.gif)|![](https://github.com/MingtaoGuo/sngan_projection_TensorFlow/blob/master/IMAGES/character1.gif)|

#### What is conditional batch normalization?
More details about 'condition', please see this repository: [Conditional Instance Normalization for n style transfer](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer)
``` python
def conditional_batchnorm(x, train_phase, scope_bn, y=None, nums_class=10):
    #Batch Normalization
    #Ioffe S, Szegedy C. Batch normalization: accelerating deep network training by reducing internal covariate shift[J]. 2015:448-456.
    with tf.variable_scope(scope_bn):
        if y == None:
            beta = tf.get_variable(name=scope_bn + 'beta', shape=[x.shape[-1]],
                                   initializer=tf.constant_initializer([0.]), trainable=True)  # label_nums x C
            gamma = tf.get_variable(name=scope_bn + 'gamma', shape=[x.shape[-1]],
                                    initializer=tf.constant_initializer([1.]), trainable=True)  # label_nums x C
        else:
            beta = tf.get_variable(name=scope_bn + 'beta', shape=[nums_class, x.shape[-1]],
                                   initializer=tf.constant_initializer([0.]), trainable=True)  # label_nums x C
            gamma = tf.get_variable(name=scope_bn + 'gamma', shape=[nums_class, x.shape[-1]],
                                    initializer=tf.constant_initializer([1.]), trainable=True)  # label_nums x C
            beta, gamma = tf.nn.embedding_lookup(beta, y), tf.nn.embedding_lookup(gamma, y)
            beta = tf.reshape(beta, [-1, 1, 1, x.shape[-1]])
            gamma = tf.reshape(gamma, [-1, 1, 1, x.shape[-1]])
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments', keep_dims=True)
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed
```
#### What is projection?
``` python
def Inner_product(global_pooled, y):
    #global_pooled: B x D,   embeded_y: B x Num_label
    H = y.shape[-1]
    W = global_pooled.shape[-1]
    V = tf.get_variable("V", [H, W], initializer=tf.truncated_normal_initializer(stddev=0.02))
    V = spectral_normalization("embed", V)
    temp = tf.matmul(y, V)
    temp = tf.reduce_sum(temp * global_pooled, axis=1)
    return temp
```
## How to use
1. Download the [ImageNet_64x64](https://patrykchrabaszcz.github.io/Imagenet32/) dataset. 
     *PS.* If necessary, you can contact me, E-mail: gmt798714378@hotmail.com, i will send a part of dataset to you :stuck_out_tongue_winking_eye:.  
2. Put the imagenet dataset into the folder 'dataset'
```
├── dataset
    ├── 1
        ├── 0.jpg
        ├── 1.jpg
        ...
    ├── 2
        ├── 0.jpg
        ├── 1.jpg
        ...
    ├── 3
        ├── 0.jpg
        ├── 1.jpg
        ...
    ...
    ├── 1000
        ├── 0.jpg
        ├── 1.jpg
        ...
├── save_para
├── save_img
├── results
├── main.py
├── networks.py
├── ops.py
├── Train.py
├── utils.py
```
3. Execute main.py
## Requirements
- python3.5
- tensorflow1.4.0
- numpy
- scipy
- pillow

### Acknowledgement
[Author's chainer code](https://github.com/pfnet-research/sngan_projection)  
### Reference
[1]. Miyato T, Koyama M. cGANs with Projection Discriminator[J]. 2018.
