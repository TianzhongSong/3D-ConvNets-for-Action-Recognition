from keras.layers import Dense, Dropout, Conv3D, Input, MaxPooling3D, Flatten, Activation
from keras.layers import concatenate, BatchNormalization, add, AveragePooling3D, GlobalAveragePooling3D
from keras.regularizers import l2
from keras.models import Model


def conv_factory(x, nb_filter, kernel=(3,3,3), weight_decay=0.005):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = Conv3D(nb_filter, kernel,
               kernel_initializer='he_normal',
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    return x


def residual_block(x, filters, drop_rate=0., weight_decay=0.005):

    x = conv_factory(x, 4 * filters, kernel=(1, 1, 1))
    if drop_rate:
        x = Dropout(drop_rate)(x)
    x = conv_factory(x, filters, kernel=(3, 3, 3))
    if drop_rate:
        x = Dropout(drop_rate)(x)
    x = Conv3D(4 * filters, (1, 1, 1),
               kernel_initializer='he_normal',
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    return x


def resnet_3d(nb_classes, input_shape, drop_rate=0., weight_decay=0.005):

    model_input = Input(shape=input_shape)

    # 112x112x8
    # stage 1 Initial convolution
    x = Conv3D(64, (3, 3, 3),
               kernel_initializer="he_normal",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(model_input)
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = MaxPooling3D((2, 2, 1), strides=(2, 2, 1), padding='same')(x)
    # 56x56x8

    # stage 2 convolution
    y = Conv3D(128, (1, 1, 1),
               kernel_initializer="he_normal",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)

    x = residual_block(x, 32, drop_rate=drop_rate)
    y = add([x, y])
    y = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(y)
    # 28x28x4

    # stage 3
    x = residual_block(y, 32, drop_rate=drop_rate)
    y = add([x, y])
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(y)
    # 14x14x2

    # stage 4
    y = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    y = Activation('relu')(y)
    y = Conv3D(256, (1, 1, 1),
                kernel_initializer='he_normal',
                padding="same",
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(y)

    x = residual_block(x, 64, drop_rate=drop_rate)
    y = add([x, y])
    x = residual_block(y, 64, drop_rate=drop_rate)
    y = add([x, y])
    y = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(y)
    # 7x7x1

    # stage 5

    x = residual_block(y, 64, drop_rate=drop_rate)
    y = add([x, y])
    x = residual_block(y, 64, drop_rate=drop_rate)
    y = add([x, y])

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(y)
    x = Activation('relu')(x)

    x = GlobalAveragePooling3D()(x)

    x = Dense(nb_classes,
              activation='softmax',
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(x)

    model = Model(inputs=model_input, outputs=x, name="resnet_3d")

    return model
