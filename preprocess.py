import os
import random
import numpy as np
import tensorflow as tf
import hyperparameter as hp
import matplotlib.pyplot as plt
from PIL import Image

class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self, data_path, task):

        self.data_path = data_path
        self.task = task

        # Dictionaries for (label index) <--> (class name)
        self.idx_to_class = {}

        # For storing list of classes
        self.classes = [""] * hp.num_classes

        # Mean and std for standardization
        self.mean = np.zeros((hp.img_size,hp.img_size, 3))
        self.std = np.ones((hp.img_size,hp.img_size, 3))
        self.calc_mean_and_std()

        # Setup data generators: feed data to the training and testing routine based on the dataset
        # task == '2': using a pretrained vgg with our own head
        # task == '3': using a pretrained mobilnet with our own head
        self.train_data = self.get_data(
            os.path.join(self.data_path, "train/"), task=='2' or task=='3', shuffle = True, augment = True)
        self.test_data = self.get_data(
            os.path.join(self.data_path, "test/"), task=='2' or task=='3', shuffle = False, augment = False)

    def calc_mean_and_std(self):
        """ Calculate mean and standard deviation of a sample of the
        training dataset for standardization.
        Input: None
        Output: None
        """

        # Get list of all images in training directory
        file_list = []
        for root, _, files in os.walk(os.path.join(self.data_path, "train/")):
            for name in files:
                if name.endswith(".jpg"):
                    file_list.append(os.path.join(root, name))

        # Shuffle filepaths and take sample of file paths
        random.shuffle(file_list)
        file_list = file_list[:hp.preprocess_sample_size]

        # Allocate space in memory for images
        data_sample = np.zeros((hp.preprocess_sample_size, hp.img_size, hp.img_size, 3))

        # Import images
        for i, file_path in enumerate(file_list):
            img = Image.open(file_path)
            img = img.resize((hp.img_size, hp.img_size))
            img = np.array(img, dtype=np.float32)
            img = np.stack([img, img, img], axis=-1)
            img /= 255.

            data_sample[i] = img

        self.mean = np.mean(data_sample, axis = 0)
        self.std = np.std(data_sample, axis = 0)


    def preprocess_fn(self, img):
        """ Preprocess function for ImageDataGenerator based on which task we're working on. """
        
        if self.task == '2':
            img = tf.keras.applications.mobilenet_v3.preprocess_input(img)
        elif self.task == '3':
            img = tf.keras.applications.vgg16.preprocess_input(img)
        else:
            img = img / 255.
            # standardize the image
            img = (img - self.mean) / self.std
        
        return img

    def get_data(self, path, is_vgg, shuffle, augment):
        """ Returns an image data generator which can be iterated
        through for images and corresponding class labels.

        Arguments:
            path - Filepath of the data being imported, such as "../data/train" or "../data/test"
            is_vgg - Boolean value indicating whether VGG or MobilNet preprocessing should be used.
            shuffle - Boolean value indicating whether the data should be randomly shuffled.
            augment - Boolean value indicating whether the data should be augmented or not.

        Returns:
            An iterable image-batch generator
        """

        if augment:
            # data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            #     preprocessing_function=self.preprocess_fn)
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn,
                rotation_range = 20,
                # brightness_range = [0.5, 1],
                zoom_range = 0.1,
                width_shift_range = 0.1,
                height_shift_range = 0.1,
                horizontal_flip=True
            )

        else:
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn)

        # VGG & MobilNet must take images of size 224x224
        img_size = 224 if is_vgg else hp.img_size

        classes_for_flow = None

        # Make sure all data generators are aligned in label indices
        if bool(self.idx_to_class):
            classes_for_flow = self.classes

        # Form image data generator from directory structure
        data_gen = data_gen.flow_from_directory(
            path,
            target_size=(img_size, img_size),
            class_mode='sparse',
            # color_mode='grayscale',
            batch_size=hp.batch_size,
            shuffle=shuffle,
            classes=classes_for_flow)

        return data_gen

# To make sure it's working fine on a specific training data
# if __name__ == '__main__':
#     dataset = Datasets("data", 1)
#     for batched_input, label in dataset.train_data:
#         print(batched_input[0].shape)
#         plt.imshow(batched_input[0])
#         plt.show()
#         raise RuntimeError()
