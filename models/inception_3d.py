from keras.layers import Dense, Dropout, Conv3D, Input, Flatten, Activation, MaxPooling3D
from keras.layers import concatenate, BatchNormalization, add, AveragePooling3D, GlobalAveragePooling3D
from keras.regularizers import l2
from keras.models import Model


def conv2d_bn(x, nb_filter, kernel=(3, 3, 3), dropout_rate=0., weight_decay=0.005):
    x = Conv3D(nb_filter, kernel,
               kernel_initializer='he_normal',
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x


def inception_3d(nb_classes, input_shape, drop_rate=0.2):
    model_input = Input(shape=input_shape)

    # 112x112x8
    # stage 1 Initial convolution
    x = conv2d_bn(model_input, 64, (3, 3, 3))
    x = MaxPooling3D((2, 2, 1), strides=(2, 2, 1), padding='same')(x)
    # 56x56x8

    # stage 1
    branch1 = conv2d_bn(x, 32, (1, 1, 1))

    branch2 = conv2d_bn(x, 32, (1, 1, 1))
    branch2 = conv2d_bn(branch2, 32, (5, 5, 3))

    branch3 = conv2d_bn(x, 32, (1, 1, 1))
    branch3 = conv2d_bn(branch3, 32, (3, 3, 3))
    branch3 = conv2d_bn(branch3, 32, (3, 3, 3))

    branch4 = AveragePooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')(x)
    branch4 = conv2d_bn(branch4, 32, (1, 1, 1))

    x = concatenate([branch1, branch2, branch3, branch4], axis=-1)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    # 28x28x4

    # stage 2
    branch1 = conv2d_bn(x, 32, (1, 1, 1))

    branch2 = conv2d_bn(x, 32, (1, 1, 1))
    branch2 = conv2d_bn(branch2, 32, (5, 5, 3))

    branch3 = conv2d_bn(x, 32, (1, 1, 1))
    branch3 = conv2d_bn(branch3, 32, (3, 3, 3))
    branch3 = conv2d_bn(branch3, 32, (3, 3, 3))

    branch4 = AveragePooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')(x)
    branch4 = conv2d_bn(branch4, 32, (1, 1, 1))

    x = concatenate([branch1, branch2, branch3, branch4], axis=-1)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    # 14x14x2

    # stage 3
    branch1 = conv2d_bn(x, 64, (1, 1, 1))

    branch2 = conv2d_bn(x, 64, (1, 1, 1))
    branch2 = conv2d_bn(branch2, 64, (5, 5, 3))

    branch3 = conv2d_bn(x, 64, (1, 1, 1))
    branch3 = conv2d_bn(branch3, 64, (7, 1, 3))
    branch3 = conv2d_bn(branch3, 64, (1, 7, 3))

    branch4 = AveragePooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')(x)
    branch4 = conv2d_bn(branch4, 64, (1, 1, 1))

    x = concatenate([branch1, branch2, branch3, branch4], axis=-1)

    branch1 = conv2d_bn(x, 64, (1, 1, 1))

    branch2 = conv2d_bn(x, 64, (1, 1, 1))
    branch2 = conv2d_bn(branch2, 64, (5, 5, 3))

    branch3 = conv2d_bn(x, 64, (1, 1, 1))
    branch3 = conv2d_bn(branch3, 64, (7, 1, 3))
    branch3 = conv2d_bn(branch3, 64, (1, 7, 3))

    branch4 = AveragePooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')(x)
    branch4 = conv2d_bn(branch4, 64, (1, 1, 1))

    x = concatenate([branch1, branch2, branch3, branch4], axis=-1)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    # 7x7x1

    # stage 4
    branch1 = conv2d_bn(x, 64, (1, 1, 1))

    branch2 = conv2d_bn(x, 64, (1, 1, 1))
    branch2 = conv2d_bn(branch2, 64, (3, 3, 1))

    branch3 = conv2d_bn(x, 64, (1, 1, 1))
    branch3 = conv2d_bn(branch3, 64, (7, 1, 1))
    branch3 = conv2d_bn(branch3, 64, (1, 7, 1))

    branch4 = AveragePooling3D(pool_size=(2, 2, 1), strides=(1, 1, 1), padding='same')(x)
    branch4 = conv2d_bn(branch4, 64, (1, 1, 1))

    x = concatenate([branch1, branch2, branch3, branch4], axis=-1)

    branch1 = conv2d_bn(x, 64, (1, 1, 1))

    branch2 = conv2d_bn(x, 64, (1, 1, 1))
    branch2 = conv2d_bn(branch2, 64, (3, 3, 1))

    branch3 = conv2d_bn(x, 64, (1, 1, 1))
    branch3 = conv2d_bn(branch3, 64, (7, 1, 1))
    branch3 = conv2d_bn(branch3, 64, (1, 7, 1))

    branch4 = AveragePooling3D(pool_size=(2, 2, 1), strides=(1, 1, 1), padding='same')(x)
    branch4 = conv2d_bn(branch4, 64, (1, 1, 1))

    x = concatenate([branch1, branch2, branch3, branch4], axis=-1)

    x = conv2d_bn(x, 256, (1, 1, 1))

    x = GlobalAveragePooling3D()(x)
    x = Dense(nb_classes,
              activation='softmax',
              kernel_regularizer=l2(0.005),
              bias_regularizer=l2(0.005))(x)
    model = Model(inputs=model_input, outputs=x)
    return model
