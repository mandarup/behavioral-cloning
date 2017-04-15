from keras import models
from keras import layers
from keras import regularizers
from keras import optimizers





def baeline(input_shape=(0,0,0)):
    crop_top_rows = int(input_shape[1] * .4)
    crop_bottom_rows = int(input_shape[1] * .05)

    model = models.Sequential()
    model.add(layers.Lambda(lambda x: x/255.-0.5,
                            input_shape=input_shape))
    model.add(layers.convolutional.Cropping2D(
        cropping=(
            (crop_top_rows, crop_bottom_rows),
            # (70,25),
            (0,0))))
    model.add(layers.Convolution2D(6,5,5,activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Convolution2D(6,5,5,activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activity_regularizer=regularizers.l1(0.01)))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(84, activity_regularizer=regularizers.l1(0.01)))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1))

    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mse', optimizer=opt)

    return model



def lenet(input_shape=None,regularizer=regularizers.l1(0.01), kernel_initializer='normal'):
    crop_top_rows = int(input_shape[1] * .35)
    crop_bottom_rows = int(input_shape[1] * .05)

    model = models.Sequential()
    model.add(layers.Lambda(lambda x: x/255.-0.5,
                            input_shape=input_shape))
    model.add(layers.convolutional.Cropping2D(
        cropping=(
            (crop_top_rows, crop_bottom_rows),
            # (70,25),
            (0,0))))
    model.add(layers.Convolution2D(6,5,5,activation='relu',kernel_initializer=kernel_initializer))
    model.add(layers.MaxPooling2D())
    model.add(layers.Convolution2D(6,5,5,activation='relu',kernel_initializer=kernel_initializer))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activity_regularizer=regularizer,kernel_initializer=kernel_initializer))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(84, activity_regularizer=regularizer,kernel_initializer=kernel_initializer))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1))

    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mse', optimizer=opt)

    return model



def lenetdeep(input_shape=(0,0,0),regularizer=regularizers.l1(0.01), kernel_initializer='normal'):
    crop_top_rows = int(input_shape[1] * .3)
    crop_bottom_rows = int(input_shape[1] * .07)

    model = models.Sequential()
    model.add(layers.Lambda(lambda x: x/255.-0.5,
                            input_shape=input_shape))
    model.add(layers.convolutional.Cropping2D(
        cropping=(
            (crop_top_rows, crop_bottom_rows),
            # (70,25),
            (0,0))))
    model.add(layers.Convolution2D(16,5,5,activation='relu',kernel_initializer=kernel_initializer))
    model.add(layers.MaxPooling2D())
    model.add(layers.Convolution2D(64,3,3,activation='relu',kernel_initializer=kernel_initializer))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D())

    model.add(layers.Convolution2D(128,3,3,activation='relu',kernel_initializer=kernel_initializer))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D())

    model.add(layers.Flatten())
    model.add(layers.Dense(200, activity_regularizer=regularizer,kernel_initializer=kernel_initializer))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(200, activity_regularizer=regularizer,kernel_initializer=kernel_initializer))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1))

    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mse', optimizer=opt)

    return model
