from ops import *

class Generator:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, train_phase, y1, y2, alpha):
        ch = 512 # paper: 1024
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            inputs = Linear("Linear", inputs, ch*2*2)
            inputs = tf.reshape(inputs, [-1, 2, 2, ch])
            inputs = G_ResBlock("ResBlock1", inputs, ch, ch, 3, train_phase, y1, y2, alpha)
            inputs = G_ResBlock("ResBlock2", inputs, ch, ch//2, 3, train_phase, y1, y2, alpha)
            inputs = G_ResBlock("ResBlock3", inputs, ch//2, ch//4, 3, train_phase, y1, y2, alpha)
            inputs = G_ResBlock("ResBlock4", inputs, ch//4, ch//8, 3, train_phase, y1, y2, alpha)
            inputs = G_ResBlock("ResBlock5", inputs, ch//8, ch//16, 3, train_phase, y1, y2, alpha)
            inputs = relu(batchnorm(inputs, train_phase, "BN_last", y1, y2, alpha))
            inputs = conv("conv_last", inputs, k_size=3, nums_in=ch//16, nums_out=3, strides=1, is_sn=False)
        return tf.nn.tanh(inputs)

    def var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

class Discriminator:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, y):
        ch = 512 # paper: 1024
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            inputs = D_ResBlock("ResBlock1", inputs, 3, 3, ch//16)
            inputs = D_ResBlock("ResBlock2", inputs, 3, ch//16, ch//8)
            inputs = D_ResBlock("ResBlock3", inputs, 3, ch//8, ch//4)
            inputs = D_ResBlock("ResBlock4", inputs, 3, ch//4, ch//2)
            inputs = D_ResBlock("ResBlock5", inputs, 3, ch//2, ch)
            inputs = D_ResBlock("ResBlock6", inputs, 3, ch, ch, False)
            inputs = relu(inputs)
            inputs = tf.reduce_sum(inputs, [1, 2])
            inner_producted = Inner_product(inputs, y)
            inputs = Linear("Linear", inputs, 1, True) + inner_producted
            return inputs

    def var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

# if __name__ == "__main__":
#     x = tf.placeholder(tf.float32, [None, 128, 128, 3])
#     z = tf.placeholder(tf.float32, [None, 100])
#     y = tf.placeholder(tf.float32, [None, 100])
#     train_phase = tf.placeholder(tf.bool)
#     G = Generator("generator")
#     D = Discriminator("discriminator")
#     fake_img = G(z, train_phase, y)
#     fake_logit = D(fake_img, y)

