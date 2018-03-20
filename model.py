import numpy as np
import os, time
import tensorflow as tf
from utils import *
from operations import *


class WGAN():
    def __init__(self, args, sess):
        self.args = args
        self.sess = sess

        self.build_model()

    def build_model(self):
        input_pipeline = image_list(self.args.dataset, self.args.training_size)
        self.real_images = read_input(input_pipeline, self.args)
        tf.summary.image('real_image', self.real_images)
        self.z = tf.placeholder(tf.float32, [None, self.args.z_dim], name='noise_z')
        tf.summary.histogram('z', self.z)

        self.g = self.generator(self.z, self.args.use_bn, reuse=False, training=True)
        tf.summary.image('generator_image', self.g)

        self.discriminator_real = self.discriminator(self.real_images, reuse=False, training=True)
        self.discriminator_fake = self.discriminator(self.g, reuse=True, training=True)
        self.generated_sample = self.generator(self.z, self.args.use_bn, reuse=True, training=False)

        self.sess.run(tf.global_variables_initializer())

        self.d_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        for v in self.d_param:
            print(v.op.name)
        self.g_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        for v in self.g_param:
            print(v.op.name)

        if self.args.improved:
            epsilon = np.random.uniform(0,1, [self.args.batch_size, 1])
            penalty_point = self.real_images * self.epsilon + self.g * (1 - self.epsilon) # [batch, 64, 64, 3]

            # Returns a list of gradients in self.d_param, mean gradient of examples in batch
            # Need index to calculate norm
            gradient = tf.gradients(self.discriminator(self.penalty_point, reuse=True), self.d_param)[0] 
            gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(self.gradient), axis=[1,2,3]))
            # Expectation over batches
            gradient_penalty = tf.reduce_mean(tf.square(gradient_norm - 1))
            self.discriminator_loss = tf.reduce_mean(self.discriminator_fake - self.discriminator_real) + self.args.penalty_hyperparam * gradient_penalty

        else:
            self.discriminator_loss = tf.reduce_mean(self.discriminator_fake - self.discriminator_real)
            tf.summary.scalar('W_distance', self.discriminator_loss)

        self.generator_loss = -tf.reduce_mean(self.discriminator_fake)
        tf.summary.scalar('G_loss', self.generator_loss)

        self.optimizer = self.get_optimizer()
        d_grads = self.optimizer.compute_gradients(self.discriminator_loss, var_list=self.d_param)
        for grad, vars in d_grads:
            if grad is not None:
                tf.summary.histogram(vars.op.name+'/gradient', grad)
        g_grads = self.optimizer.compute_gradients(self.generator_loss, var_list=self.g_param)
        for grad, vars in g_grads:
            if grad is not None:
                tf.summary.histogram(vars.op.name+'/gradient', grad)
        # Discriminator gradient
        self.d_optimizer = self.optimizer.apply_gradients(d_grads)
        # Generator gradient
        self.g_optimizer = self.optimizer.apply_gradients(g_grads)
        self.clipping_op = [disc_param.assign(tf.clip_by_value(disc_param, -self.args.clip, self.args.clip)) for disc_param in self.d_param]

        self.saver = tf.train.Saver()

    def generator(self, z, use_batchnorm, reuse=False, training=True):
        with tf.variable_scope('generator') as scope:
            if reuse:
                scope.reuse_variables()

            z_flatten = linear(z, self.args.target_size // 16 * self.args.target_size // 16 * self.args.final_dim * 8, name='llinear')
            z_reshaped = tf.reshape(z_flatten, [self.args.batch_size, self.args.target_size // 16, self.args.target_size // 16, self.args.final_dim * 8])
            if use_batchnorm:
                de1 = deconv2d(z_reshaped, output_shape=[self.args.batch_size, self.args.target_size // 8, self.args.target_size // 8, self.args.final_dim * 4], name='gen_batch1')
                de1_result = relu_with_batch(de1, training, name='relu_batch1')
                de2 = deconv2d(de1_result, output_shape=[self.args.batch_size, self.args.target_size // 4, self.args.target_size // 4, self.args.final_dim *2], name='gen_batch2')
                de2_result = relu_with_batch(de2, training, name='relu_batch2')
                de3 = deconv2d(de2_result, output_shape=[self.args.batch_size, self.args.target_size // 2, self.args.target_size // 2, self.args.final_dim], name='gen_batch3')
                de3_result = relu_with_batch(de3, training, name='relu_batch3')
                de4 = deconv2d(de3_result, output_shape=[self.args.batch_size, self.args.target_size, self.args.target_size, self.args.num_channels], name='gen_batch4')

                return tf.nn.tanh(de4, name='generator_result')
            else:
                de1 = deconv2d(z_reshaped, output_shape=[self.args.batch_size, self.args.target_size // 8, self.args.target_size // 8, self.args.final_dim *4], name='gen_wo_batch1')
                de1_result = relu(de1, name='relu1')
                de2 = deconv2d(de1_result, output_shape=[self.args.batch_size, self.args.target_size // 4, self.args.target_size // 4, self.args.final_dim * 2], name='gen_wo_batch2')
                de2_result = relu(de2, name='relu2')
                de3 = deconv2d(de2_result, output_shape=[self.args.batch_size, self.args.target_size // 2, self.args.target_size // 2, self.args.final_dim], name='gen_wo_batch3')
                de3_result = relu(de3, name='relu3')
                de4 = deconv2d(de3_result, output_shape=[self.args.batch_size, self.args.target_size, self.args.target_size, self.args.num_channels], name='gen_wo_batch4')

                return tf.nn.tanh(de4, name='generator_result')

    # In WGAN, discriminator or critic does not output probability anymore(That`s why it is called as critic) So just output logits
    def discriminator(self, real_or_fake, reuse=False, training=True):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            cn1 = conv2d(real_or_fake, self.args.final_dim, name='disc1')
            cn1_result = leaky_relu(cn1, name='lrelu_batch1') # First layer skips batch norm
            cn2 = conv2d(cn1_result, self.args.final_dim * 2, name='disc2')
            cn2_result = leaky_relu_with_batch(cn2, training, name='lrelu_batch2')
            cn3 = conv2d(cn2_result, self.args.final_dim * 4, name='disc3')
            cn3_result = leaky_relu_with_batch(cn3, training, name='lrelu_batch3')
            cn4 = conv2d(cn3_result, self.args.final_dim * 8, name='disc4')
            cn4_result = leaky_relu_with_batch(cn4, training, name='lrelu_batch4')
            cn4_result_reshaped = tf.reshape(cn4_result, [self.args.batch_size, -1])
            disc_flattend = linear(cn4_result_reshaped, 1, name='linear')

            return disc_flattend

    def get_optimizer(self):
        if self.args.improved:
            return tf.train.AdamOptimizer(self.args.learning_rate)
        else:
            return tf.train.RMSPropOptimizer(self.args.learning_rate)

    def train(self):
        start_time = time.time()
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.args.log_dir, self.sess.graph)
        self.sample_z = np.random.uniform(-1, 1, [self.args.showing_height * self.args.showing_width, self.args.z_dim])
        sample_dir = os.path.join(self.args.sample_dir, self.model_dir)
        print(sample_dir)
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)

        self.sess.run(tf.global_variables_initializer())

        if self.load():
            print('Checkpoint loaded')
        else:
            print('Checkpoint load failed')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        try:
           for epoch in range(self.train_count, self.args.num_epoch):
               print('Epoch %d start' % (epoch+1))
               for train_critic_step in range(self.args.train_critic):
                   batch_z = np.random.uniform(-1,1,[self.args.batch_size, self.args.z_dim])
                   disc_loss, _ = self.sess.run([self.discriminator_loss, self.d_optimizer], feed_dict={self.z:batch_z})
                   if not self.args.improved:
                        self.sess.run(self.clipping_op)
                   print('Critic train %d, loss : %3.4f' % (train_critic_step+1, disc_loss))
               batch_z = np.random.uniform(-1, 1, [self.args.batch_size, self.args.z_dim])
               gen_loss, sum_op, _ = self.sess.run([self.generator_loss, self.summary_op, self.g_optimizer], feed_dict={self.z:batch_z})
               self.summary_writer.add_summary(sum_op, epoch+1)

               if np.mod(epoch+1, self.args.save_step) == 0:
                   G_sample = self.sess.run(self.generated_sample, feed_dict={self.z:self.sample_z})
                   if self.args.improved:
                       save_image(G_sample, [self.args.showing_height, self.args.showing_width], os.path.join(sample_dir, 'train_{:2d}steps_improved.jpg'.format(epoch+1)))
                   else:
                       save_image(G_sample, [self.args.showing_height, self.args.showing_width], os.path.join(sample_dir, 'train_{:2d}steps.jpg'.format(epoch+1)))
                   self.save(epoch+1)
               print('Epoch %d, discriminator loss : %3.4f, generator loss : %3.4f, duration time : %3.4f' % (epoch+1, disc_loss, gen_loss, time.time() - start_time))

        except tf.errors.OutOfRangeError:
            print('Epoch limited')
        except KeyboardInterrupt:
            print('End training')
        finally:
            coord.request_stop()
            coord.join(threads=threads)

    @property
    def model_dir(self):
        if not self.args.improved:
            return '{}_batch_{}_z_dim_{}'.format(self.args.batch_size, self.args.z_dim, 'CelebA')
        if self.args.improved:
            return '{}_batch_{}_z_dim_{}_improved'.format(self.args.batch_size, self.args.z_dim, 'CelebA')

    def save(self, global_step):
        model_name='Wasserstein_gan'
        checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=global_step)
        print('Checkpoint saved at %d steps' % global_step)

    def load(self):
        checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.train_count = int(ckpt_name.split('-')[-1])
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print('Checkpoint loaded at %d steps' % self.train_count)
            return True
        else:
            self.train_count = 0
            return False
