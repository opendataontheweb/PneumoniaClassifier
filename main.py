import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

datagenerator = ImageDataGenerator(rescale=1. / 255)

train = datagenerator.flow_from_directory('chest_xray/train/', class_mode='binary', target_size=(227, 227))

test = datagenerator.flow_from_directory('chest_xray/test/', class_mode='binary', target_size=(227, 227))

val = datagenerator.flow_from_directory('chest_xray/val/', class_mode='binary', target_size=(227, 227))

inp = keras.layers.Input(shape=(227, 227, 3))
x = keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu')(inp)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
x = keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
x = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(4096, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(4096, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
output = keras.layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs=inp, outputs=output, name='AlexNetXRAY')

print(model.summary())

model.compile(loss=keras.losses.BinaryCrossentropy, optimizer=keras.optimizers.SGD(), metrics=['accuracy'])

model.fit(train, epochs=15, validation_data=val)

model.save('alexnet_xray')

print(model.evaluate(test))
