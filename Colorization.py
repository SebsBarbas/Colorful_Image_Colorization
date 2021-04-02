import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D, Reshape, Multiply, MaxPooling2D, Cropping2D, \
    UpSampling2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras import Model


class Colorization(Model):
    def __init__(self, shape):
        super(Colorization, self).__init__()

        self.conv1_1 = Conv2D(64, [3, 3], activation = 'relu', strides = (2, 2), padding = 'same', data_format = 'channels_last', input_shape = shape, name = 'conv1_1')
        self.conv1_2 = Conv2D(64, [3, 3], activation = 'relu', padding = 'same', name = 'conv1_2')
        self.conv1_3 = Conv2D(64, [3, 3], activation = 'relu', name = 'conv1_3')
        self.norm1 = BatchNormalization()

        # second layer
        self.conv2_1 = Conv2D(128, [3, 3], activation = 'relu', padding = 'same', strides = (2, 2), name = 'conv2_1')
        self.conv2_2 = Conv2D(128, [3, 3], activation = 'relu', padding = 'same', name = 'conv2_2')
        self.conv2_3 = Conv2D(128, [3, 3], activation = 'relu', padding = 'same', name = 'conv2_3')
        self.norm2 = BatchNormalization()


        # Third layer

        self.conv3_1 = Conv2D(256, [3, 3], activation = 'relu',  padding = 'same', strides = (2, 2), name = 'conv3_1')
        self.conv3_2 = Conv2D(256, [3, 3], activation = 'relu',  padding = 'same', name = 'conv3_2')
        self.conv3_3 = Conv2D(256, [3, 3], activation = 'relu',  padding = 'same', name = 'conv3_3')
        self.norm3 = BatchNormalization()

        # Fourth layer

        self.conv4_1 = Conv2D(512, [3, 3], activation = 'relu', padding = 'same', name = 'conv4_1')
        self.conv4_2 = Conv2D(512, [3, 3], activation = 'relu', padding = 'same', name = 'conv4_2')
        self.conv4_3 = Conv2D(512, [3, 3], activation = 'relu', padding = 'same', name = 'conv4_3')
        self.norm4 = BatchNormalization()

        #Fifth layer

        self.conv5_1 = Conv2D(512, [3, 3], activation = 'relu', padding = 'same', name = 'conv5_1')
        self.conv5_2 = Conv2D(512, [3, 3], activation = 'relu', padding = 'same', name = 'conv5_2')
        self.conv5_3 = Conv2D(512, [3, 3], activation = 'relu', padding = 'same', name = 'conv5_3')
        self.norm5 = BatchNormalization()


        # Sixth layer

        self.conv6_1 = Conv2D(512, [3, 3], activation = 'relu', padding = 'same', name = 'conv6_1')
        self.conv6_2 = Conv2D(512, [3, 3], activation = 'relu', padding = 'same', name = 'conv6_2')
        self.conv6_3 = Conv2D(512, [3, 3], activation = 'relu', padding = 'same', name = 'conv6_3')
        self.norm6 = BatchNormalization()

        #Seventh layer

        self.conv7_1 = Conv2D(512, [3, 3], activation = 'relu', padding = 'same', name = 'conv7_1')
        self.conv7_2 = Conv2D(512, [3, 3], activation = 'relu', padding = 'same', name = 'conv7_2')
        self.conv7_3 = Conv2D(512, [3, 3], activation = 'relu', padding = 'same', name = 'conv7_3')
        self.norm7 = BatchNormalization()

        #Eigth layer

        self.conv8_1 = Conv2DTranspose(256, [2, 2], activation = 'relu', strides = 2, padding = 'same', name = 'upconv8')
        self.conv8_2 = Conv2D(313, [3, 3], activation = 'softmax', padding = 'same', name = 'conv8')

    def call(self, input, training = False):
        # First layer
        x = self.conv1_1(input)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.norm1(x, training = training)


        # Second layer
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.norm2(x, training = training)

        # Third layer
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.norm3(x, training = training)

        # Fourth layer
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.norm4(x, training = training)

        # Fifth layer
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.norm5(x, training = training)

        # Sixth layer
        x = self.conv6_1(x)
        x = self.conv6_2(x)
        x = self.conv6_3(x)
        x = self.norm6(x, training = training)

        # Seventh layer
        x = self.conv7_1(x)
        x = self.conv7_2(x)
        x = self.conv7_3(x)
        x = self.norm7(x, training = training)

        # Eigth layer
        x = self.conv8_1(x)
        x = self.conv8_2(x)

        return x


if __name__ == '__main__':
    model = Colorization((256, 256, 1))
    #tf.keras.utils.plot_model(model, to_file = 'Unet_plot.png', show_shapes = True, show_layer_names = True)
    model.build((None, 256, 256, 1))
    tf.keras.utils.plot_model(model, to_file = 'Unet_plot.png', show_shapes = True, show_layer_names = True)
    a = 1











