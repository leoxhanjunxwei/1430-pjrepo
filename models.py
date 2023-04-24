import tensorflow as tf
from keras.layers import \
       Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, ReLU, LeakyReLU

import hyperparameter as hp

from keras.applications.vgg16 import VGG16
# Can also import MobileNetV2, or MobileNetV3Large
from keras.applications import MobileNetV3Small


class YourModel(tf.keras.Model):
    def __init__(self):
        super(YourModel, self).__init__()

        # Define an optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
        # self.optimizer = tf.keras.optimizers.SGD(learning_rate=hp.learning_rate, momentum=hp.momentum)
        
        input_shape = (hp.batch_size, hp.img_size, hp.img_size, 3)
        self.architecture = [
              # Add layers here separated by commas.
              Conv2D(filters = 128, kernel_size = (5, 5), padding="same", input_shape = input_shape),
              BatchNormalization(),
              ReLU(),
              
              Conv2D(filters = 128, kernel_size = (5, 5), padding="same"),
              BatchNormalization(),
              ReLU(),
              MaxPool2D(pool_size = (4, 4)),

              Conv2D(filters = 256, kernel_size = (5, 5), padding="same"),
              BatchNormalization(),
              ReLU(),
              
              Conv2D(filters = 256, kernel_size = (5, 5), padding="same"),
              BatchNormalization(),
              ReLU(),
              MaxPool2D(pool_size = (4, 4)),
              
              Conv2D(filters = 512, kernel_size = (5, 5), padding="same"),
              BatchNormalization(),
              ReLU(),
            
              Conv2D(filters = 512, kernel_size = (5, 5), padding="same"),
              BatchNormalization(),
              ReLU(),
              MaxPool2D(pool_size = (4, 4)),
              
              Flatten(),
              Dense(units = 400, activation = "relu"),
              Dropout(0.2),
              Dense(units = hp.num_classes, activation = "softmax")   
        ]

    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)
        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)
        loss_fn = scce(labels, predictions)
        return loss_fn

class MobileNetModel(tf.keras.Model):
    def __init__(self):
        super(MobileNetModel, self).__init__()

        # Hyperparameters
        self.learning_rate = 5e-4
        # Momentum on the gradient (for momentum-based optimizer)
        self.momentum = 0.01
        
        # Initialize the optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        # self.optimizer = tf.keras.optimizers.SGD(learning_rate=hp.learning_rate, momentum=self.momentum)

        self.mobilenet = MobileNetV3Small(include_top=False, weights='imagenet')

        # Freeze the convolutional base
        self.mobilenet.trainable = False
        
        # Add a classification head
        self.head = [Flatten(),
                     Dense(200),
                     BatchNormalization(),
                     LeakyReLU(),
                     Dense(hp.num_classes, activation='softmax')]
        self.head = tf.keras.Sequential(self.head, name="vgg_head")

    def call(self, x):
        x = self.mobilenet(x)
        x = self.head(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """

        scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)
        loss_fn = scce(labels, predictions)
        return loss_fn
    
    
class VGGModel(tf.keras.Model):
    def __init__(self):
        super(VGGModel, self).__init__()
        
        # Hyperparameters
        self.learning_rate = 5e-4
        # Momentum on the gradient (for momentum-based optimizer)
        self.momentum = 0.01
        
        # Initialize the optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        # self.optimizer = tf.keras.optimizers.SGD(learning_rate=hp.learning_rate, momentum=self.momentum)

        # Create the base model of VGG16
        self.vgg16 = VGG16(include_top=False, weights = 'imagenet')

        # Freeze the convolutional base
        self.vgg16.trainable = False
        
        # Add a classification head
        self.head = [Flatten(),
                     Dense(200),
                     BatchNormalization(),
                     LeakyReLU(),
                     Dense(hp.num_classes, activation='softmax')]
        self.head = tf.keras.Sequential(self.head, name="vgg_head")

    def call(self, x):
        """ Passes the image through the network. """

        x = self.vgg16(x)
        x = self.head(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """
        
        scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)
        loss_fn = scce(labels, predictions)
        return loss_fn
