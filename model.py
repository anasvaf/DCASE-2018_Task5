import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib import distributions as dist
from tensorflow.contrib import slim
import numpy as np

TOL = 1e-5
log2pi = tf.constant(np.log(2 * np.pi), tf.float32)

def sample_gaussian(latent_dim, mean, variance, batch_size):
    epsilon = tf.random_normal([batch_size, latent_dim])
 
    return mean + tf.sqrt(variance) * epsilon
 
 
def standard_gaussian_likelihood(inputs):
    input_dim = inputs.get_shape().as_list()[1]
 
    return -(log2pi * (input_dim / 2.0)) - \
           tf.reduce_sum(tf.square(inputs) / 2.0, 1)
 
 
def gaussian_log_likelihood(inputs, mean, variance):
    input_dim = inputs.get_shape().as_list()[1]
    variance = tf.sqrt(variance)

    return -(log2pi * (input_dim / 2.0)) - \
           tf.reduce_sum((tf.square(inputs - mean) / (2 * variance)) + (variance / 2.0), 1)
 
 
def bernoulli_log_likelihood(inputs, mean):
    return tf.reduce_sum((inputs * tf.log(TOL + mean)) + ((1 - inputs) * tf.log(TOL + 1 - mean)), 1)
 
 
def compute_loss(inf_mean_list, inf_var_list, gen_mean_list, gen_var_list,
                 q_log_discrete, log_px, batch_size):
    gaussian_div = []
 
    for mean0, var0, mean1, var1 in zip(inf_mean_list, inf_var_list,
                                        reversed(gen_mean_list), reversed(gen_var_list)):
        kl_gauss = dist.kl_divergence(dist.MultivariateNormalDiag(mean0, var0),
                                      dist.MultivariateNormalDiag(mean1, var1))
        gaussian_div.append(kl_gauss)
 
    kl_gauss = tf.reshape(tf.concat(gaussian_div, axis=0), [batch_size, len(gaussian_div)])
    kl_dis = dist.kl_divergence(dist.OneHotCategorical(logits=q_log_discrete),
                                dist.OneHotCategorical(logits=tf.log(tf.ones_like(q_log_discrete) * 1/10)))
    mean_KL = tf.reduce_mean(tf.reduce_sum(kl_gauss, axis=1) + kl_dis)
    mean_rec = tf.reduce_mean(log_px)
    loss = tf.reduce_mean(log_px - 0.5 * ((tf.reduce_sum(kl_gauss, axis=1) + kl_dis)))

    return loss, mean_rec, mean_KL


def mlp(inputs, is_training, layer1_units, layer2_units):
    aff1 = layers.dense(inputs, layer1_units, activation=None)
    bn1 = layers.batch_normalization(aff1, center=True, scale=True,
                                     training=is_training)
    l1 = tf.nn.relu(bn1)

    aff2 = layers.dense(l1, layer2_units, activation=None)
    bn2 = layers.batch_normalization(aff2, center=True, scale=True,
                                     training=is_training)
    l2 = tf.nn.relu(bn2)

    return l2
 

def conv_net(inputs, conv1_kernels, kernel_size1, pool_size1,
             conv2_kernels, kernel_size2, pool_size2):
    # *** WARNING: HARDCODED SHAPES, CHANGE ACCORDINGLY ***
    in_layer = tf.reshape(inputs, [-1, 28, 28, 1])
    
    # 1st conv layer
    conv1 = layers.conv2d(inputs=in_layer,
                          filters=conv1_kernels,
                          kernel_size=kernel_size1,
                          padding='same',
                          activation=tf.nn.relu)

    # # 1st pooling layer
    # pool1 = layers.max_pooling2d(inputs=conv1, 
    #                              pool_size=pool_size1, 
    #                              strides=2)

    # 2nd conv layer
    conv2 = layers.conv2d(inputs=conv1,
                          filters=conv2_kernels,
                          kernel_size=kernel_size2,
                          padding='same',
                          activation=tf.nn.relu)

    # # 2nd pooling layer
    # pool2 = layers.max_pooling2d(inputs=conv2, 
    #                              pool_size=pool_size2, 
    #                              strides=2)
    fl = layers.Flatten()(conv2)
    return fl


def deconv_net(inputs):
    # *** WARNING: HARDCODED SHAPES, CHANGE ACCORDINGLY ***
    dense = layers.dense(inputs, units=7*7*16, activation=tf.nn.relu)
    # input(dense)
    dense = tf.reshape(dense, [16, 7, 7, 16])

    # 1st deconv layer
    deconv1 = layers.conv2d_transpose(dense,
                                      filters=32, 
                                      kernel_size=[5, 5],
                                      padding='same',
                                      activation=tf.nn.relu)
    # input(deconv1)
    # 2nd deconv layer
    deconv2 = layers.conv2d_transpose(deconv1,
                                      filters=16, 
                                      kernel_size=[5, 5],
                                      padding='same',
                                      activation=tf.nn.relu)
    # input(deconv2)
    out = layers.conv2d_transpose(deconv2, filters=16, kernel_size=[3, 3], padding='same')
    return out
    


def inference_net(observations, is_training, hidden1, hidden2, n_out):
    # inf_net = mlp(observations, is_training, hidden1, hidden2)
    inf_net = conv_net(observations, 16, [5, 5], [2, 2], 32, [5, 5], [2, 2])
    mu = slim.fully_connected(inf_net, n_out, activation_fn=None)
    sigma_sq = TOL + slim.fully_connected(inf_net, n_out, activation_fn=tf.nn.softplus)
 
    return mu, sigma_sq
 
 
def generator_net(observations, is_training, hidden1, hidden2,
                  n_out, likelihood='bernoulli'):
    # gen_net = mlp(observations, is_training, hidden1, hidden2)
    gen_net = deconv_net(observations)
    gen_net = tf.reshape(gen_net, [16, 784])

    if likelihood == 'bernoulli':
        probs = slim.fully_connected(gen_net, n_out, activation_fn=tf.nn.sigmoid)
 
        return probs
    elif likelihood == 'gaussian':
        mu = slim.fully_connected(gen_net, n_out, activation_fn=None)
        sigma_sq = TOL + slim.fully_connected(gen_net, n_out, activation_fn=tf.nn.softplus)
 
        return mu, sigma_sq
    else:
        raise ValueError("Unsupported likelihood function.")
 
 
def inference_model(observations, is_training, latent_layer_dims, tau, batch_size, nn_layers):
    samples = [observations]
    q_log_likelihoods = []
    mean_list = []
    var_list = []
 
    for i in range(len(latent_layer_dims)):
        mu, sigma_sq = inference_net(samples[i], is_training, nn_layers[0], nn_layers[1], latent_layer_dims[i])
       
        sample = sample_gaussian(latent_layer_dims[i], mu, sigma_sq, batch_size)
        sample_ll = gaussian_log_likelihood(sample, mu, sigma_sq)
 
        mean_list.append(mu)
        var_list.append(sigma_sq)
        q_log_likelihoods.append(sample_ll)
        samples.append(sample)
 
    logits_qy = slim.fully_connected(mlp(samples[-1], is_training, nn_layers[0], nn_layers[1]), 10, activation_fn=None)
    q_y = dist.RelaxedOneHotCategorical(tau, logits=logits_qy)
 
    sample = q_y.sample()
    samples.append(sample)
    q_log_likelihoods.append(tf.log(sample + TOL))
    samples.remove(samples[0])
 
    return samples, mean_list, var_list, q_log_likelihoods
 
 
def generative_model(observations, samples, is_training, latent_layer_dims, nn_layers):
    samples = list(reversed(samples))
    latent_layer_dims = list(reversed(latent_layer_dims))

    mu, sigma_sq = generator_net(samples[0], is_training, nn_layers[0], nn_layers[1], latent_layer_dims[0], 'gaussian')
    mean_list = [mu]
    var_list = [sigma_sq]
    p_lls = []
    p_gen = None

    # reconstruction of training samples
    for i in range(1, len(samples) - 1):
        mu, sigma_sq = generator_net(samples[i], is_training, nn_layers[0], nn_layers[1], latent_layer_dims[i], 'gaussian')
        p_lls.append(dist.MultivariateNormalDiag(mu, sigma_sq))

        mean_list.append(mu)
        var_list.append(sigma_sq)

    probs = generator_net(samples[-1], is_training, nn_layers[0], nn_layers[1], observations.get_shape().as_list()[1],
                          likelihood='bernoulli')
    p_x = bernoulli_log_likelihood(observations, probs)

    # generation of novel samples
    sample_gen = tf.random_uniform([16], maxval=11, dtype=tf.int32)
    sample_gen = tf.one_hot(sample_gen, 10)
    mu_gen, sigma_sq_gen = generator_net(sample_gen, is_training, nn_layers[0], nn_layers[1], latent_layer_dims[0], 'gaussian')
    gen_samples = [dist.MultivariateNormalDiag(mu_gen, sigma_sq_gen).sample()]

    for i in range(1, len(latent_layer_dims) - 1):
        mu, sigma_sq = generator_net(samples[i], is_training, nn_layers[0], nn_layers[1], latent_layer_dims[i], 'gaussian')
        gen_samples.append(dist.MultivariateNormalDiag(mu, sigma_sq).sample())

    probs = generator_net(gen_samples[-1], is_training, nn_layers[0], nn_layers[1], observations.get_shape().as_list()[1],
                        likelihood='bernoulli')
    p_gen = dist.Bernoulli(probs=probs).sample()

    return probs, p_gen, p_x, mean_list, var_list
