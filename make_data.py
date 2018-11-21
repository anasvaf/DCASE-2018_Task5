import tensorflow as tf
import numpy as np


def make_mnist_dataset(batch_size, shuffle=True, include_labels=True):
    """Loads the MNIST data set and returns the relevant
    iterator along with its initialization operations.
    """

    # load the data
    train, test = tf.keras.datasets.mnist.load_data()
    
    # binarize and reshape the data sets
    temp_train = train[0]
    temp_train = (temp_train > 0.5).astype(np.float32).reshape(temp_train.shape[0], 784)
    train = (temp_train, train[1])

    temp_test = test[0]
    temp_test = (temp_test > 0.5).astype(np.float32).reshape(temp_test.shape[0], 784)
    test = (temp_test, test[1])

    # prepare Dataset objects
    if include_labels:
        train_set = tf.data.Dataset.from_tensor_slices(train).repeat().batch(batch_size)
        test_set = tf.data.Dataset.from_tensor_slices(test).repeat(1).batch(batch_size)
    else:
        train_set = tf.data.Dataset.from_tensor_slices(train[0]).repeat().batch(batch_size)
        test_set = tf.data.Dataset.from_tensor_slices(test[0]).repeat(1).batch(batch_size)

    if shuffle:
        train_set = train_set.shuffle(buffer_size=int(0.5*train[0].shape[0]), 
                                      seed=123)

    # make the iterator
    iter = tf.data.Iterator.from_structure(train_set.output_types,
                                           train_set.output_shapes)
    data = iter.get_next()

    # create initialization ops
    train_init = iter.make_initializer(train_set)
    test_init = iter.make_initializer(test_set)

    return train_init, test_init, data


def make_dataset(train_data, valid_data, test_data, batch_size, shuffle=True):
    """Loads a user-provided data set and returns the relevant
    iterator along with its initialization operations.
    """

    # prepare Dataset objects
    train_set = tf.data.Dataset.from_tensor_slices(train_data).repeat().batch(batch_size)
    valid_set = tf.data.Dataset.from_tensor_slices(valid_data).repeat().batch(batch_size)
    test_set = tf.data.Dataset.from_tensor_slices(test_data).repeat(1).batch(batch_size)

    if shuffle:
        train_set = train_set.shuffle(buffer_size=int(0.5*train_data[0].shape[0]),
                                      seed=123)

    # make the iterator
    iter = tf.data.Iterator.from_structure(train_set.output_types,
                                           train_set.output_shapes)
    data = iter.get_next()

    # create initialization ops
    train_init = iter.make_initializer(train_set)
    valid_init = iter.make_initializer(valid_set)
    test_init = iter.make_initializer(test_set)

    return train_init, valid_init, test_init, data
