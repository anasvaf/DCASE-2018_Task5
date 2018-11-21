from make_data import make_mnist_dataset
from model import inference_model, generative_model, compute_loss
from visualization import plot_pca, show_image
import tensorflow as tf
import numpy as np


def build_graph(x, y, learning_rate, is_training, latent_var_dim, tau, batch_size, inf_layers, gen_layers):
    latent_vars, inf_mean_list, inf_var_list, q_lls = inference_model(x, is_training, latent_var_dim, tau, batch_size, inf_layers)
    img_vec, gen_imgs, log_px, gen_mean_list, gen_var_list = generative_model(x, latent_vars, is_training, latent_var_dim, gen_layers)

    elbo, mean_rec, mean_KL = compute_loss(inf_mean_list, inf_var_list,
                                           gen_mean_list, gen_var_list,
                                           q_lls[-1], log_px, batch_size)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(-elbo)

    return elbo, mean_rec, mean_KL, optimizer, latent_vars, img_vec, gen_imgs


def train_model(training_iters, test_set_sz, learning_rate, batch_size, 
                is_training, latent_dim, tau, inf_layers, gen_layers,
                save_path=None):
    train_init, test_init, next_batch = make_mnist_dataset(batch_size, include_labels=True)
    ops = build_graph(next_batch[0], next_batch[1], learning_rate, is_training, 
                      latent_dim, tau, batch_size, inf_layers, gen_layers)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_init)

        print("Training...")
        for i in range(1, training_iters):
            loss, rec, kl, _ = sess.run(ops[:4])

            # don't spam the output channel,
            # print every 500 iterations
            if i % 500 == 0:
                print("Iteration", i,  
                    "\tloss: {:.3f}".format(loss), 
                    "\tmean reconstruction: {:.3f}".format(rec), 
                    "\tmean KL: {:.3f}".format(kl))
        saver.save(sess, './saved_models/model.ckpt')

        return test_init, ops, next_batch


def test_model(trained_ops, test_set_sz, learning_rate, batch_size, 
               latent_dim, tau, inf_layers, gen_layers,
               restore_path):
    test_init = trained_ops[0]
    ops = trained_ops[1]
    next_batch = trained_ops[2]
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        mean_loss = 0
        mean_rec = 0
        mean_kl = 0
        test_iters = int(test_set_sz/batch_size)
        lat = None
        lab = None
        gen_imgs = None
        rec_imgs = None

        saver.restore(sess, './saved_models/model.ckpt')
        sess.run(test_init)
        
        print("\nTesting...")
        for i in range(test_iters):
            loss, rec, kl, z, x_hat, x_gen, y = sess.run((ops[0], ops[1], ops[2], ops[4], ops[5], ops[6], next_batch[1]))
            mean_loss += loss/test_iters
            mean_rec += rec/test_iters
            mean_kl += kl/test_iters
            if lat is None:
                lat = z[0]
                lab = y
            else:
                lat = np.append(lat, z[0], axis=0)
                lab = np.append(lab, y, axis=0)
            
            # reconstruct samples and generate new ones
            if i == test_iters - 3 or i == test_iters - 2 or i == test_iters - 1:
                if gen_imgs is None:
                    rec_imgs = x_hat.reshape((batch_size * 28, 28))
                    gen_imgs = x_gen.reshape((batch_size * 28, 28))
                else:
                    rec_imgs = np.append(rec_imgs, x_hat.reshape((batch_size * 28, 28)), axis=1)
                    gen_imgs = np.append(gen_imgs, x_gen.reshape((batch_size * 28, 28)), axis=1)

        show_image(gen_imgs, './images/generated.png')
        show_image(rec_imgs, './images/reconstructed.png')
        print("\tloss: {:.3f}".format(mean_loss), 
              "\tmean reconstruction: {:.3f}".format(mean_rec), 
              "\tmean KL: {:.3f}".format(mean_kl))
        plot_pca(10, lat, lab, './figures/test_latent_space')
