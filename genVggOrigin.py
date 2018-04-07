import tensorflow as tf
import numpy as np
import depth_net_utils as utils
import matplotlib.pyplot as plt


class Model:
    def __init__(self, params_path, img_dem_row, img_dem_col):
        self.img_dem_row = img_dem_row
        self.img_dem_col = img_dem_col
        self.current_stage = ''
        self.weights = utils.get_weights(params_path + "/weights")
        # print(self.weights)
        self.biases = utils.get_weights(params_path + "/biases")
        # print(self.biases)
        self.output = 0
        self.session = None
        self.model_output = 0
        self.vars_initialized = False
        self.model_input = tf.placeholder(tf.float32, shape=(None, self.img_dem_row, self.img_dem_col, None), name="x")
        self.__build_model()

    def __get_weights(self, name):
        current_stage_weights = self.weights[self.current_stage]
        return tf.constant(current_stage_weights[name], name='weights')

    def __get_biases(self, name):
        current_stage_biases = self.biases[self.current_stage]
        return tf.constant(current_stage_biases[name], name='biases')

    def __conv_layer(self, _x, name):
        with tf.variable_scope(self.current_stage):
            kernel = self.__get_weights(name + '_W')
            bias = self.__get_biases(name + '_b')

            conv = tf.nn.conv2d(_x, kernel, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, bias)

            return tf.nn.relu(bias)

    def __conv_block(self, block_input, layer_names):
        self.output = block_input
        for i in layer_names:
            self.output = self.__conv_layer(self.output, i)
        return self.output

    def __max_pool(self, bottom, name):
        with tf.variable_scope(self.current_stage):
            return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def __fc_layer(self, _x, name):
        with tf.variable_scope(self.current_stage):
            shape = _x.get_shape().as_list()
            weights = self.__get_weights(name)
            biases = self.__get_biases(name)

            die = 1
            for dem_d in shape[1:]:
                die *= dem_d
            _x = tf.reshape(_x, [-1, die])

        return tf.nn.bias_add(tf.matmul(_x, weights), biases)

    def __build_vgg_model(self):
        """
        Model Architecture = conv_block1 -> max_pool -> conv_block2 -> max_pool -> conv_block3 -> max_pool ->
        conv_block4 -> max_pool -> conv_block5 -> FC1 -> FC2
        """
        self.current_stage = "vgg"
        vgg_input = tf.image.resize_bilinear(self.model_input, size=(112, 150))

        conv1 = self.__conv_block(vgg_input, ['conv1_1', 'conv1_2'])
        pool1 = self.__max_pool(conv1, 'pool1')

        conv2 = self.__conv_block(pool1, ['conv2_1', 'conv2_2'])
        pool2 = self.__max_pool(conv2, 'pool2')

        conv3 = self.__conv_block(pool2, ['conv3_1', 'conv3_2'])
        pool3 = self.__max_pool(conv3, 'pool3')

        conv4 = self.__conv_block(pool3, ['conv4_1', 'conv4_2', 'conv4_3'])
        pool4 = self.__max_pool(conv4, 'pool4')

        conv5 = self.__conv_block(pool4, ['conv5_1', 'conv5_2', 'conv5_3'])
        pool5 = self.__max_pool(conv5, 'pool5')

    def __build_model(self):
        self.__build_vgg_model()

    def __init_tensorflow_session(self):
        if self.session is None:
            self.session = tf.Session()

    def __initialize_variables(self):
        if not self.vars_initialized:
            self.session.run(tf.initialize_all_variables())

model = Model('./vgg_raw',128,128)

# vgg/conv1_1/kernel:0
# <tf.Variable 'ConvNet/conv1/kernel:0' shape=(5, 5, 1, 32) dtype=float32_ref>
# <tf.Variable 'ConvNet/conv1/bias:0' shape=(32,) dtype=float32_ref>

for name in sorted(model.weights['vgg']):
    _ = tf.Variable(model.weights['vgg'][name] ,name='vgg/'+name)

for name in sorted(model.biases['vgg']):
    _ = tf.Variable(model.biases['vgg'][name] ,name='vgg/'+name)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saveVar = tf.global_variables()
saver = tf.train.Saver(saveVar)
save_path = saver.save(sess, "./savedModel/vgg_original.ckpt")
for t in saveVar:
    print(t.name, t.shape)