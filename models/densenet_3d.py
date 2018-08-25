from keras.layers import Dense, Dropout, Conv3D, Input, MaxPool3D, Flatten, Activation
from keras.layers import concatenate, BatchNormalization, AveragePooling3D, GlobalAveragePooling3D
from keras.regularizers import l2
from keras.models import Model


def conv_factory(x, nb_filter, kernel=(3,3,3), dropout_rate=0., weight_decay=0.005):
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


def transition_layer():
    pass


def dense_block(x, growth_rate, internal_layers=4,
               dropout_rate=0., weight_decay=0.005):
    list_feat = []
    list_feat.append(x)
    x = conv_factory(x, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
    list_feat.append(x)
    x = concatenate(list_feat, axis=-1)
    for i in range(internal_layers - 1):
        x = conv_factory(x, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
        list_feat.append(x)
        x = concatenate(list_feat, axis=-1)
    return x


def densenet_3d(nb_classes, input_shape, weight_decay=0.005, dropout_rate=0.2):

    model_input = Input(shape=input_shape)

    # 112x112x8
    # stage 1 Initial convolution
    x = conv_factory(model_input, 64)
    x = MaxPool3D((2, 2, 1), strides=(2, 2, 1), padding='same')(x)
    # 56x56x8

    # stage 2
    x = dense_block(x, 32, internal_layers=4,
                             dropout_rate=dropout_rate)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    x = conv_factory(x, 128, (1, 1, 1), dropout_rate=dropout_rate)
    # 28x28x4

    # stage 3
    x= dense_block(x, 32, internal_layers=4,
                   dropout_rate=dropout_rate)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    x = conv_factory(x, 128, (1, 1, 1), dropout_rate=dropout_rate)

    # 14x14x2

    # stage 4
    x = dense_block(x, 64, internal_layers=4,
                   dropout_rate=dropout_rate)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    x = conv_factory(x, 256, (1, 1, 1), dropout_rate=dropout_rate)

    # 7x7x1

    # stage 5
    x = dense_block(x, 64, internal_layers=4,
                   dropout_rate=dropout_rate)

    x = conv_factory(x, 256, (1, 1, 1), dropout_rate=dropout_rate)

    x = GlobalAveragePooling3D()(x)
    x = Dense(nb_classes,
              activation='softmax',
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(x)

    model = Model(inputs=model_input, outputs=x, name="densenet_3d")

    return model
