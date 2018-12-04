import tensorflow as tf



def batchnorm(x, train_phase, scope_bn, y1=None, y2=None, alpha=1.):
    #Batch Normalization
    #Ioffe S, Szegedy C. Batch normalization: accelerating deep network training by reducing internal covariate shift[J]. 2015:448-456.
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
    _v = tf.stop_gradient(_v)
    _u = tf.stop_gradient(_u)
    sigma = tf.reduce_sum(tf.matmul(_u, W) * _v)
    return sigma, _u, _v

def spectral_normalization(name, W, Ip=1):
    u = tf.get_variable(name + "_u", [1, W.shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)  # 1 x ch
    W_mat = tf.transpose(tf.reshape(W, [-1, W.shape[-1]]))
    sigma, _u, _ = max_singular_value(W_mat, u, Ip)
    with tf.control_dependencies([tf.assign(u, _u)]):
        W_sn = W / sigma
    return W_sn


def conv(name, inputs, k_size, nums_in, nums_out, strides, is_sn=True):
    # nums_in = inputs.shape[-1]
    kernel = tf.get_variable(name+"W", [k_size, k_size, nums_in, nums_out], initializer=tf.truncated_normal_initializer(stddev=0.02))
    bias = tf.get_variable(name+"B", [nums_out], initializer=tf.constant_initializer(0.))
    if is_sn:
        return tf.nn.conv2d(inputs, spectral_normalization(name, kernel), [1, strides, strides, 1], "SAME") + bias
    else:
        return tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], "SAME") + bias

def deconv(name, inputs, k_size, nums_out, strides):
    nums_in = inputs.shape[-1]
    output_B = tf.shape(inputs, 0)
    output_H = tf.shape(inputs, 1) * strides
    output_W = tf.shape(inputs, 2) * strides
    kernel = tf.get_variable(name + "W", [k_size, k_size, nums_out, nums_in], initializer=tf.truncated_normal_initializer(stddev=0.02))
    bias = tf.get_variable(name + "B", [nums_out], initializer=tf.constant_initializer(0.))
    return tf.nn.conv2d_transpose(inputs, kernel, [output_B, output_H, output_W, nums_out], [1, strides, strides, 1], "SAME") + bias

def relu(inputs):
    return tf.nn.relu(inputs)

def avg_pool(inputs, k_size=3, strides=2, padding="SAME"):
    return tf.nn.avg_pool(inputs, [1, k_size, k_size, 1], [1, strides, strides, 1], padding)

def D_ResBlock(name, inputs, k_size, nums_in, nums_out, is_down=True):
    #inputs: B x H x W x C_in
    temp = inputs
    inputs = relu(inputs)
    inputs = conv("conv1_" + name, inputs, k_size, nums_in, nums_out, 1, True)#inputs: B x H/2 x W/2 x C_out
    inputs = relu(inputs)
    inputs = conv("conv2_" + name, inputs, k_size, nums_out, nums_out, 1, True)#inputs: B x H/2 x W/2 x C_out
    if is_down:
        inputs = avg_pool(inputs)
        down_sampling = conv("down_sampling_"+name, temp, 1, nums_in, nums_out, 1, True)#down_sampling: B x H x W x C_out
        down_sampling = avg_pool(down_sampling)
        outputs = inputs + down_sampling
    else:
        outputs = inputs + temp
    return outputs

def G_ResBlock(name, inputs, nums_in, nums_out, k_size, train_phase, y1=None, y2=None, alpha=1.0, is_up=True):
    # inputs: B x H x W x C_in
    H = inputs.shape[1]
    W = inputs.shape[2]
    temp = inputs
    inputs = batchnorm(inputs, train_phase, "BN1_"+name, y1, y2, alpha)
    inputs = relu(inputs)
    if is_up:
        inputs = tf.image.resize_nearest_neighbor(inputs, [H * 2, W * 2])#inputs: B x H*2 x W*2 x C_in
    inputs = conv("conv1_"+name, inputs, k_size, nums_in, nums_out, 1, False)
    inputs = batchnorm(inputs, train_phase, "BN2_"+name, y1, y2, alpha)
    inputs = relu(inputs)
    inputs = conv("conv2_"+name, inputs, k_size, nums_out, nums_out, 1, False)#inputs: B x H*2 x W*2 x C_out
    if is_up:
        temp = tf.image.resize_nearest_neighbor(temp, [H * 2, W * 2])#inputs: B x H*2 x W*2 x C_in
        temp = conv("up_sampling_"+name, temp, 1, nums_in, nums_out, 1, False)#inputs: B x H*2 x W*2 x C_out

    return inputs + temp


def Linear(name, inputs, nums_out, is_sn=False):
    nums_in = inputs.shape[-1]
    W = tf.get_variable("W_" + name, [nums_in, nums_out], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b = tf.get_variable("B_" + name, [nums_out], initializer=tf.constant_initializer(0.))
    if is_sn:
        return tf.matmul(inputs, spectral_normalization(name, W)) + b
    else:
        return tf.matmul(inputs, W) + b

def Inner_product(global_pooled, y):
    #global_pooled: B x D,   embeded_y: B x Num_label
    H = y.shape[-1]
    W = global_pooled.shape[-1]
    V = tf.get_variable("V", [H, W], initializer=tf.truncated_normal_initializer(stddev=0.02))
    V = spectral_normalization("embed", V)
    temp = tf.matmul(y, V)
    temp = tf.reduce_sum(temp * global_pooled, axis=1)
    return temp

def Hinge_Loss(fake_logits, real_logits):
    D_loss = tf.reduce_mean(tf.maximum(0., 1 - real_logits)) + \
             tf.reduce_mean(tf.maximum(0., 1 + fake_logits))
    # D_loss = -tf.reduce_mean(tf.log(real_logits + 1e-14)) - tf.reduce_mean(tf.log(1 - fake_logits + 1e-14))
    # G_loss = -tf.reduce_mean(tf.log(fake_logits + 1e-14))
    # D_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
    G_loss = -tf.reduce_mean(fake_logits)
    return D_loss, G_loss




