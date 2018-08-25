from keras.layers import Dense, Dropout, Conv3D, Input, MaxPool3D, Flatten, Activation
from keras.layers import concatenate, BatchNormalization, add, AveragePooling3D, GlobalAveragePooling3D
from keras.regularizers import l2
from keras.models import Model


def c3d_model():
    input_shape = (112, 112, 16, 3)
    weight_decay = 0.005
    nb_classes = 101

    inputs = Input(input_shape)
    x = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(inputs)
    x = MaxPool3D((2, 2, 1), strides=(2, 2, 1), padding='same')(x)

    x = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x4 = Flatten()(x)
    x3 = Dense(2048, activation='relu', kernel_regularizer=l2(weight_decay))(x4)
    # x = Dropout(0.5)(x)
    x2 = Dense(2048, activation='relu', kernel_regularizer=l2(weight_decay))(x3)
    # x = Dropout(0.5)(x)
    x1 = Dense(nb_classes, kernel_regularizer=l2(weight_decay))(x2)
    # x = Activation('softmax')(x)
    out = concatenate([x1, x2, x3, x4], axis=-1)
    model = Model(inputs, out)
    return model


def conv_factory(x, nb_filter, dropout_rate=0., weight_decay=0.005):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = Conv3D(nb_filter, (3, 3, 3),
               kernel_initializer='he_normal',
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x


def denseblock(x, growth_rate, strides=(1, 1, 1), internal_layers=4,
               dropout_rate=0., weight_decay=0.005):
    x = Conv3D(growth_rate, (3, 3, 3),
               kernel_initializer='he_normal',
               padding="same",
               strides=strides,
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    list_feat = []
    list_feat.append(x)
    for i in range(internal_layers - 1):
        x = conv_factory(x, growth_rate, dropout_rate, weight_decay)
        list_feat.append(x)
        x = concatenate(list_feat, axis=-1)
    x = Conv3D(internal_layers * growth_rate, (1, 1, 1),
               kernel_initializer='he_normal',
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    return x


def Residual_DenseNet(nb_classes, input_shape, weight_decay=0.005, dropout_rate=0.2, extract_feat=False):
    internal_layers = 3

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
    x = MaxPool3D((2, 2, 1), strides=(2, 2, 1), padding='same')(x)
    # 56x56x8
    # stage 2 convolution
    x = Conv3D(96, (3, 3, 3),
               kernel_initializer="he_normal",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    y = Activation('relu')(x)
    y = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(y)
    # 28x28x4

    x1 = Conv3D(96, (1, 1, 4),
                kernel_initializer='he_normal',
                padding="valid",
                strides=(4, 4, 4),
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(y)

    # stage 3
    x = denseblock(y, 32, internal_layers=internal_layers,
                   dropout_rate=dropout_rate)

    y = add([x, y])
    y = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(y)
    # 14x14x2

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(y)
    x = Activation('relu')(x)

    x2 = Conv3D(96, (1, 1, 2),
                kernel_initializer='he_normal',
                padding="valid",
                strides=(2, 2, 2),
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(x)

    # stage 4
    x = denseblock(x, 32, internal_layers=internal_layers,
                   dropout_rate=dropout_rate)
    y = add([x, y])

    x3 = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(y)
    # 7x7x1
    # stage 5
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x3)
    x = Activation('relu')(x)

    x = denseblock(x, 32, internal_layers=internal_layers,
                   dropout_rate=dropout_rate)
    y = add([x, x3])

    # concat y, x3, x2, x1
    x = concatenate([y, x3, x2, x1], axis=-1)
    # 7x7x1
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)

    x = Conv3D(512, (1, 1, 1),
               kernel_initializer='he_normal',
               padding="same",
               strides=(1, 1, 1),
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling3D()(x)
    if not extract_feat:
        x = Dense(nb_classes,
                  activation='softmax',
                  kernel_regularizer=l2(weight_decay),
                  bias_regularizer=l2(weight_decay))(x)

    DRN = Model(input=[model_input], output=[x], name="DRN")

    return DRN
