"""
Image size for CNN. VGG16 must take in image size of 224, so that is hard-coded elsewhere.
"""
img_size = 48

"""
The number of facial emotion classes.
"""
num_classes = 7

"""
Number of training examples per batch.
"""
batch_size = 50

"""
Sample size for calculating the mean and standard deviation of the
training data. This many images will be randomly seleted to be read
into memory temporarily.
"""
preprocess_sample_size = 500

"""
Number of epochs from which we decrease or increase our learning rate.
"""
num_ep_decrease = 15

"""
Number of epochs. If you experiment with more complex networks you
might need to increase this. Likewise if you add regularization that
slows training.
"""
num_epochs = 30

"""
Maximum number of weight files to save to checkpoint directory. If
set to a number <= 0, then all weight files of every epoch will be
saved. Otherwise, only the weights with highest accuracy will be saved.
"""
max_num_weights = 5