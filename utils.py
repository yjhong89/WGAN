import tensorflow as tf
import numpy as np
import time, os, sys, glob
import argparse
import scipy.misc

def image_list(image_dir, training_size):
    # Check image directory existency
    if not os.path.exists(image_dir):
        print('Image directory %s not exists' % image_dir)
        return None
    file_type_extended = ('jpg', 'jpeg', 'png')
    file_list = []
    for path, _dir, files in os.walk(image_dir):
        print('path : %s' %path)
        print('dir : %s ' %_dir)
        print('files : %s' % files)
        #file_dir_path = os.path.join(os.path.basename(image_dir), path)
        for _file in files:
            if _file.split('.')[-1] in file_type_extended:
                file_list.append(os.path.join(os.path.abspath(image_dir), _file))
            else:
                pass
    if len(file_list) == 0:
        print('No image files')
        return None
    else:
        print('Number of files %d' % len(file_list))
        print('Training size %d' % training_size)
        '''
        shuffle : Boolean. If true, the strings are randomly shuffled within each epoch
        capacity : An integer, Sets the queue capacity maximum
        num_epoch : If not specified string_input_producer can cycle through the strings in inp
        FIFOQUEUE + QUEUERUNNER'''
        file_list_queue = tf.train.string_input_producer(file_list, capacity=training_size, shuffle=True) # Need to change 'queue'  from ''list'
        return file_list_queue

def read_files_preprocess(file_list_queue, args):
    image_reader = tf.WholeFileReader()
    key, value = image_reader.read(file_list_queue) # Returns both string scalar tensor
    uint8_image = tf.image.decode_jpeg(value, channels=args.num_channels) # Returns of type uint8 with [height, width, channels], # tf.image.decode_image not working
    # CelebA data original dimension : [218,178,3]
    cropped_image = tf.cast(tf.image.crop_to_bounding_box(uint8_image, offset_height=50, offset_width=35, target_height=args.input_size ,target_width=args.input_size), tf.float32)
    cropped_image_4d = tf.expand_dims(cropped_image, 0)
    resized_image = tf.image.resize_bilinear(cropped_image_4d, size=[args.target_size, args.target_size])
    input_image = tf.squeeze(resized_image, squeeze_dims=[0])
    return input_image

def read_input(file_list_queue, args):
    inp_img = read_files_preprocess(file_list_queue, args)
    num_preprocess_threads = 4
    min_queue_examples = int(0.5*args.num_examples_per_epoch)
    print('Shuffling inputs %s' %(inp_img.get_shape()))
    '''    
    This function adds : 1. A shuffling queue into which tensors form tensor list are enqueued. 2. A dequeue_many operation to craete batches from the queue, 3. A QueueRunner to enqueue the tensors form tensor list
    shuffle_batch constructs a RandomShuffleQueue and proceeds to fill with a QueueRunner. The queue accumulates examples sequentialy until in contains bach_size + min_after_dequeue examples are present.
    It then selects batch_size random element from the queue to return The value actually returned by shuffle_batch is the result of a dequeue_may call on the RandomShuffleQueue
    enqueue_many argument set as False -> input shape will be [x,y,z], output will be [batch, x,y,z]
    Reference from http://stackoverflow.com/questions/36334371/tensorflow-batching-input-queues-then-changing-the-queue-source'''
    #  tensors part should be iterable so [] needed(tensor list)
    input_image = tf.train.shuffle_batch([inp_img], batch_size=args.batch_size, num_threads=num_preprocess_threads, capacity=min_queue_examples+3*args.batch_size,  min_after_dequeue=min_queue_examples)
    input_image = input_image / 127.5  - 1
    return input_image

def save_image(imgs, size, path):
    print('Save images')
    #print(imgs)
    #print(imgs.shape)
    height, width = imgs.shape[1], imgs.shape[2]
    merged_image = np.zeros([size[0]*height, size[1]*width, imgs.shape[3]])
    for image_index, img in enumerate(imgs):
        j = image_index % size[1]
        i = image_index // size[1]
        merged_image[i*height:i*height+height, j*width:j*width+width, :] = img
    merged_image += 1
    merged_image *= 127.5
    merged_image = np.clip(merged_image, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, merged_image)

if __name__ == "__main__":
	file_queue = image_list('../CelebA', 100000)	
	a = read_input(file_queue)
