import tensorflow as tf
import numpy as np
import time, os, argparse
from model import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, default=200000)
    parser.add_argument('--clip', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='../CelebA')
    parser.add_argument('--learning_rate', type=float, default=0.00005)
    parser.add_argument('--train_critic', type=int, default=5)
    parser.add_argument('--is_training', type=str2bool, default='yes')
    parser.add_argument('--improved', type=str2bool, default='no')
    parser.add_argument('--use_bn', type=str2bool, default='no')
    parser.add_argument('--num_examples_per_epoch', type=int, default=10)
    parser.add_argument('--penalty_hyperparam', type=int, default=10)
    parser.add_argument('--training_size', type=int, default=10000)
    parser.add_argument('--input_size', type=int, default=108)
    parser.add_argument('--target_size', type=int, default=64)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--final_dim', type=int, default=64)
    parser.add_argument('--showing_height', type=int, default=8)
    parser.add_argument('--showing_width', type=int, default=8)
    parser.add_argument('--sample_dir', type=str, default='./samples')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint')
    parser.add_argument('--log_dir', type=str, default='./tensorboard_log')
    parser.add_argument('--save_step', type=int, default=200)

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.mkdir(args.sample_dir)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        wgan = WGAN(args, sess)
        if args.is_training:
            print('Training starts')
            wgan.train()
        else:
            print('Test')
            wgan.generator_test()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n' '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Not expected boolean type')

if __name__ == "__main__":
    main()
