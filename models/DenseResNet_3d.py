from keras.layers import Dense, Dropout, Conv3D, Input, MaxPool3D, Activation, add
from keras.layers import concatenate, BatchNormalization, GlobalAveragePooling3D
from keras.regularizers import l2
from keras.models import Model


def conv_factory(x, nb_filter, kernel=(3, 3, 3), strides=(1, 1, 1),
                 padding='same', dropout_rate=0., weight_decay=0.005):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = Conv3D(nb_filter, kernel,
               kernel_initializer='he_normal',
               padding=padding,
               strides=strides,
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x


def dense_block(x, growth_rate, internal_layers=4,
               dropout_rate=0., weight_decay=0.005):
    x = conv_factory(x, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
    list_feat = []
    list_feat.append(x)
    for i in range(internal_layers - 1):
        x = conv_factory(x, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
        list_feat.append(x)
        x = concatenate(list_feat, axis=-1)
    return x


def dense_resnet_3d(nb_classes, input_shape, weight_decay=0.005, dropout_rate=0.2):

    model_input = Input(shape=input_shape)

    # 112x112x8
    # stage 1 Initial convolution
    x = Conv3D(64, (3, 3, 3),
               kernel_initializer="he_normal",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(model_input)
    x = MaxPool3D((2, 2, 1), strides=(2, 2, 1), padding='same')(x)
    # 56x56x8

    y = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    y = Activation('relu')(y)
    y = Conv3D(128, (1, 1, 1),
               kernel_initializer="he_normal",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(y)

    # stage 2
    x = dense_block(x, 32, internal_layers=4,
                             dropout_rate=dropout_rate)
    x = add([x, y])
    y = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    # 28x28x4

    # stage 3
    x= dense_block(y, 32, internal_layers=4,
                   dropout_rate=dropout_rate)
    x = add([x, y])
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    # 14x14x2

    y = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    y = Activation('relu')(y)
    y = Conv3D(256, (1, 1, 1),
               kernel_initializer="he_normal",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(y)
    x1 = conv_factory(x, 128, (1, 1, 2), (2, 2, 2), padding='valid')

    # stage 4
    x = dense_block(x, 64, internal_layers=4,
                   dropout_rate=dropout_rate)
    x = add([x, y])
    y = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    # 7x7x1

    x2 = conv_factory(y, 128, (1, 1, 1), (1, 1, 1), padding='same')

    # stage 5
    x = dense_block(y, 64, internal_layers=4,
                   dropout_rate=dropout_rate)
    x = add([x, y])
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)

    x = concatenate([x, x2, x1], axis=-1)
    x = Conv3D(512, (1, 1, 1),
               kernel_initializer='he_normal',
               padding="same",
               strides=(1, 1, 1),
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling3D()(x)
    x = Dense(nb_classes,
              activation='softmax',
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(x)

    model = Model(inputs=model_input, outputs=x)

    return model
