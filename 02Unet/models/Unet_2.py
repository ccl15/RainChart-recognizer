import tensorflow as tf
from tensorflow.keras import layers
'''
6 layers, 2 conv, 32 filters
'''


class Convolution_operator(layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size=(3,3), kernel_initializer='he_normal',
                                   activation=layers.LeakyReLU(), padding='same')
        self.batch_norm1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, kernel_size=(3,3), kernel_initializer='he_normal',
                                   activation=layers.LeakyReLU(),  padding='same')
        self.batch_norm2 = layers.BatchNormalization()

    def __call__(self, x, training):
        x = self.batch_norm1(self.conv1(x), training=training)
        x = self.batch_norm2(self.conv2(x), training=training)
        return x
    
class DownSample(layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.conv_AF = Convolution_operator(filters)
        self.MaxPool = layers.MaxPooling2D((2, 2), padding='same')
     
    def __call__(self, x, training):
        x = self.conv_AF(x, training)
        MaxP = self.MaxPool(x)
        return x, MaxP
    
class UpSample(layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.conv_t = layers.Conv2DTranspose(filters, kernel_size=(3,3), strides=(2,2), padding='same')
        self.conv_AF = Convolution_operator(filters)
     
    def __call__(self, x, copy, training):
        x = self.conv_t(x)
        if not x.get_shape() == copy.get_shape():
            sh = copy.get_shape()
            x = tf.image.crop_to_bounding_box(x, 0, 0, sh[1], sh[2])
        connect = layers.concatenate([x, copy])
        out = self.conv_AF(connect, training)
        return out


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        depth = 32
        self.down_layers = [
            DownSample(depth*1),
            DownSample(depth*2),
            DownSample(depth*4),
            DownSample(depth*8),
            DownSample(depth*16)]
        self.lowest_conv = Convolution_operator(depth*32)
        self.up_layers = [
            UpSample(depth*16),
            UpSample(depth*8),
            UpSample(depth*4),
            UpSample(depth*2),
            UpSample(depth*1)]
        self.output_conv = layers.Conv2D(1, (3,3), kernel_initializer='he_normal',
                                         activation='sigmoid',  padding='same')

    def U_net(self, x, training):
        # copy of each layer
        layer_stack = []
        # Down sampleing => (copy, output which passed to next level)
        for down in self.down_layers:
            copy ,x = down(x, training=training)
            layer_stack.append(copy)
        # lowest level 
        x = self.lowest_conv(x, training=training)
        # Up sampleing
        for up in self.up_layers:
            copy = layer_stack.pop()
            x = up(x, copy, training=training)
        # convolution to output
        x = self.output_conv(x)
        return x
    
    def __call__(self, input_data, training=False):
        Output_image = self.U_net(input_data, training)
        return Output_image 
    
