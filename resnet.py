from keras.models import Model
from keras.models import Sequential 
from sklearn.utils import shuffle
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout
from keras.layers import merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D

def resnet(input_shape=(256,256,3), nb_classes=5, optimizer='Adam', loss='binary_crossentropy'):
    resnet_model = ResNet50(weights='imagenet')
    inp = Input(shape=input_shape)
    output_resnet = resnet_model(inp)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), init='he_normal', name='conv1')(output_resnet)
    x = BatchNormalization(axis=1, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
#   x = AveragePooling2D((7, 7))(x)
    x = Flatten()(x)
    x = Dense(nb_classes, init='he_normal', activation='softmax', name='fc10')(x)

    model = Model(inp, x)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
        )
    return model