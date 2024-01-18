import numpy as np
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from tensorflow.keras import Sequential
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50

np.random.seed(1337)


class Eye_models:
    def __init__(self):
        self.hand_made_model = Sequential()
        self.train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        self.test_datagen = ImageDataGenerator(rescale=1. / 255)
        self.training_set = self.train_datagen.flow_from_directory('Data/Eye/train', target_size=(128, 128),
                                                                   class_mode='categorical')
        self.training_set_resnet = self.train_datagen.flow_from_directory('Data/Eye/train', target_size=(224, 224),
                                                                          class_mode='categorical')
        self.test_set = self.test_datagen.flow_from_directory('Data/Eye/val', target_size=(128, 128),
                                                              class_mode='categorical')
        self.test_set_resnet = self.test_datagen.flow_from_directory('Data/Eye/val', target_size=(224, 224),
                                                                     class_mode='categorical')
        self.label_map = self.training_set.class_indices
        self.vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
        self.resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    def hand_made_model_setup(self):
        self.hand_made_model.add(Convolution2D(32, 3, 3, input_shape=(128, 128, 3), activation='relu'))
        self.hand_made_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.hand_made_model.add(Convolution2D(16, 3, 3, activation='relu'))
        self.hand_made_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.hand_made_model.add(Convolution2D(8, 3, 3, activation='relu'))
        self.hand_made_model.add(MaxPooling2D(pool_size=(1, 1)))

        self.hand_made_model.add(Flatten())
        self.hand_made_model.add(Dense(units=128, activation='relu'))
        self.hand_made_model.add(Dropout(rate=0.5))
        self.hand_made_model.add(Dense(units=4, activation='softmax'))

        self.hand_made_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return self.hand_made_model

    def VGG16_model_setup(self):
        for layer in self.vgg16.layers:
            layer.trainable = False

        flatten_layer = layers.Flatten()
        dense_layer_1 = layers.Dense(50, activation='relu')
        dense_layer_2 = layers.Dense(20, activation='relu')
        prediction_layer = layers.Dense(4, activation='softmax')

        model = models.Sequential([self.vgg16,
                                   flatten_layer,
                                   dense_layer_1,
                                   dense_layer_2,
                                   prediction_layer])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def ResNet50_model_setup(self):
        flatten_layer = layers.Flatten()
        for layer in self.resnet.layers:
            layer.trainable = False

        model = models.Sequential([self.resnet,
                                   flatten_layer,
                                   Dense(512, activation='relu', input_dim=(224, 224, 3)),
                                   Dropout(0.3),
                                   Dense(512, activation='relu'),
                                   Dropout(0.3),
                                   Dense(4, activation='softmax')])
        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(learning_rate=2e-5),
                      metrics=['accuracy'])
        return model

    def model_train(self, model, flag):
        if flag == 0:
            model.fit(self.training_set, epochs=100)
        elif flag == 1:
            model.fit(self.training_set_resnet, epochs=100)

    def model_testing(self, model, flag):
        if flag == 0:
            return model.evaluate(self.test_set)
        elif flag == 1:
            return model.evaluate(self.test_set_resnet)
