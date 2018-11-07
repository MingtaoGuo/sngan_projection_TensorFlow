# sngan_projection_TensorFlow
Simply implementing the paper: cGANs with Projection Discriminator

## Introduction
SNGAN with projection discriminator implemented by TensorFlow. The paper [cGANs with Projection Discriminator](https://arxiv.org/pdf/1802.05637v1.pdf)

![](https://github.com/MingtaoGuo/sngan_projection_TensorFlow/blob/master/IMAGES/projection.jpg)

#### What is spectral norm?
``` python
def _l2normalize(v, eps=1e-12):
    return v / tf.sqrt(tf.reduce_sum(tf.square(v)) + eps)


def max_singular_value(W, u=None, Ip=1):
    if u is None:
        u = tf.get_variable("u", [1, W.shape[-1]], initializer=tf.random_normal_initializer(), trainable=False) #1 x ch
    _u = u
    _v = 0
    for _ in range(Ip):
        _v = _l2normalize(tf.matmul(_u, W), eps=1e-12)
        _u = _l2normalize(tf.matmul(_v, W, transpose_b=True), eps=1e-12)
    sigma = tf.reduce_sum(tf.matmul(_u, W) * _v)
    return sigma, _u, _v

def spectral_normalization(name, W, Ip=1):
    u = tf.get_variable(name + "_u", [1, W.shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)  # 1 x ch
    W_mat = tf.transpose(tf.reshape(W, [-1, W.shape[-1]]))
    sigma, _u, _ = max_singular_value(W_mat, u, Ip)
    with tf.control_dependencies([tf.assign(u, _u)]):
        W_sn = W / sigma
    return W_sn
```
#### What is conditional batch normalization?
More details about 'condition', please see this repository: [Conditional Instance Normalization](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer)
``` python
def batchnorm(x, train_phase, scope_bn, y1=None, y2=None, alpha=1.):
    with tf.variable_scope(scope_bn):
        if y1 == None:
            beta = tf.get_variable(name=scope_bn + 'beta', shape=[x.shape[-1]],
                                   initializer=tf.constant_initializer([0.]), trainable=True)  # label_nums x C
            gamma = tf.get_variable(name=scope_bn + 'gamma', shape=[x.shape[-1]],
                                    initializer=tf.constant_initializer([1.]), trainable=True)  # label_nums x C
        else:
            beta = tf.get_variable(name=scope_bn + 'beta', shape=[y1.shape[-1], x.shape[-1]],
                                   initializer=tf.constant_initializer([0.]), trainable=True)  # label_nums x C
            gamma = tf.get_variable(name=scope_bn + 'gamma', shape=[y1.shape[-1], x.shape[-1]],
                                    initializer=tf.constant_initializer([1.]), trainable=True)  # label_nums x C
            beta1 = tf.matmul(y1[:1, :], beta)
            gamma1 = tf.matmul(y1[:1, :], gamma)
            beta2 = tf.matmul(y2[:1, :], beta)
            gamma2 = tf.matmul(y2[:1, :], gamma)
            beta = beta1 * alpha + beta2 * (1 - alpha)
            gamma = gamma1 * alpha + gamma2 * (1 - alpha)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
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
## Results
### More results are under training ......
As shown in below is trained about 10000 iterations with batch size of 64.

![](https://github.com/MingtaoGuo/sngan_projection_TensorFlow/blob/master/IMAGES/generate.jpg)

#### Consecutive category morphing with fixed z:
|cat2human|cat2human|zi2zi|zi2zi|
|-|-|-|-|
|![](https://github.com/MingtaoGuo/sngan_projection_TensorFlow/blob/master/IMAGES/1.gif)|![](https://github.com/MingtaoGuo/sngan_projection_TensorFlow/blob/master/IMAGES/2.gif)|![](https://github.com/MingtaoGuo/sngan_projection_TensorFlow/blob/master/IMAGES/character.gif)|![](https://github.com/MingtaoGuo/sngan_projection_TensorFlow/blob/master/IMAGES/character1.gif)|
### Acknowledgement
[Author's chainer code](https://github.com/pfnet-research/sngan_projection)  
### Reference
[1]. Miyato T, Koyama M. cGANs with Projection Discriminator[J]. 2018.
