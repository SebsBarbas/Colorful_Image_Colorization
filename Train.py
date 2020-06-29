import os
import sys
import time
import pickle
import random
import numpy as np
from Colorization import *
import glob
import cv2
import tensorflow as tf
from scipy import spatial
import datetime

files = "./test2014/*.jpg"
imges = [f for f in glob.glob(files)]


def load_data(ab_2_class, ab_points):
    train_images = []
    out_images = []
    for file in imges:

        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2LAB)

        if len(train_images) == 0:
            test = cv2.resize(img, (256, 256))
            train_images = [cv2.resize(img, (256, 256))[:, :, 0] / 255]
        else:
            train_images.append(cv2.resize(img, (256, 256))[:, :, 0] / 255)

        output = cv2.resize(img, (64, 64))[:, :, 1:]
        #out_im = np.zeros((output.shape[0], output.shape[1], 313))
        out_im = np.zeros((output.shape[0], output.shape[1]))

        for i in range(out_im.shape[0]):
            for j in range(out_im.shape[1]):
                ab_val = output[i, j] // 10
                ab_val = ab_val * 10
                if str([ab_val[0], ab_val[1]]) not in ab_2_class.keys():
                    key =  ab_points[spatial.KDTree(ab_points).query(ab_val)[1]]
                    ab_val = key
                pos = ab_2_class[str([int(ab_val[0]), int(ab_val[1])])]
                #out_im[i, j, pos] = 1
                out_im[i, j] = pos

        if len(out_images) == 0:
            out_images = [out_im]
        else:
            out_images.append(out_im)

    train = train_images[128:]
    train_out = out_images[128:]
    test = train_images[:128]
    test_out = out_images[:128]

    return train, train_out, test, test_out

def decode(ab_image, maps, T=0.35):

    ab_second = tf.multiply(ab_image, 1/T)
    ab_second = tf.keras.activations.softmax(ab_second, -1)

    max_pos_soft = tf.math.argmax(ab_second, -1)
    max_pos = tf.math.argmax(ab_image, -1)
    shapes = tf.shape(max_pos)
    shapes = shapes.numpy()

    decoding_soft = np.zeros((shapes[0], shapes[1], 2))
    decoding = np.zeros((shapes[0], shapes[1], 2))

    for i in range(shapes[0]):
        for j in range(shapes[1]):
            val_soft = tf.gather_nd(max_pos_soft, (i, j)).numpy()
            tup_soft = maps[str(val_soft)]
            decoding_soft[i, j, 0] = tup_soft[0]
            decoding_soft[i, j, 1] = tup_soft[1]

            val = tf.gather_nd(max_pos, (i, j)).numpy()
            tup = maps[str(val)]
            decoding[i, j, 0] = tup[0]
            decoding[i, j, 1] = tup[1]



    return decoding, decoding_soft

cat_loss = tf.keras.losses.CategoricalCrossentropy()
spars_loss = tf.keras.losses.SparseCategoricalCrossentropy()

def loss_color(conv_out, gt_ab):

    flat_conv_out = tf.reshape(conv_out, [-1, 313])
    flat_gt_ab = tf.reshape(gt_ab, [-1, 313])
    #loss = cat_loss(flat_gt_ab, flat_conv_out)/conv_out.get_shape()[0]
    loss = spars_loss(flat_gt_ab, flat_conv_out) / conv_out.get_shape()[0]

    return loss


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training = True)
        print(predictions.get_shape())
        loss =loss_color(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training = False)
    t_loss = loss_color(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
model = Colorization((256, 256))
batch_size = 128
train_loss = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'train_accuracy')

test_loss = tf.keras.metrics.Mean(name = 'test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'test_accuracy')
test_top1 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name = 'test_1_accuracy')
test_top5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name = 'test_5_accuracy')




if __name__ == "__main__":

    save_model_dir = ".\checkpoints"
    infile = open('dictionaries.p', 'rb')
    list_dict = pickle.load(infile)
    infile.close()
    map_ab_2_class = list_dict[0]
    map_class_2_ab = list_dict[1]
    ab_points = list_dict[2]

    train, train_out, test, test_out = load_data(map_ab_2_class, ab_points)

    train_set = tf.data.Dataset.from_tensor_slices((train, train_out)).batch(batch_size)
    test_set = tf.data.Dataset.from_tensor_slices((test, test_out)).batch(batch_size)
    i = 0



    checkpoint_dir = os.path.join(save_model_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer = optimizer, model = model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep = 3)
    #checkpoint.restore(manager.latest_checkpoint)
    #if manager.latest_checkpoint:
    #    print("Restaurado de {}".format(manager.latest_checkpoint))
    #else:
    #    print("Inicializando desde cero")

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    for images, labels in train_set:
        train_step(images, labels)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=i+1)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=i+1)


        if i % 5 == 0:
            for img, lbl in test_set:
                test_step(img, lbl)
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step = i + 1)
                tf.summary.scalar('accuracy', test_accuracy.result(), step = i + 1)

            save_path = manager.save()

        i = i + 1

        template = 'Epoch {}, Train Loss: {}, Train Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(i+1, train_loss.result(), train_accuracy.result()*100, \
                                test_loss.result(), test_accuracy.result()*100))
        template = 'Top1 Error: {}, Top5 Error: {}'
        print(template.format((1 - test_top1.result())*100,\
                                    (1 - test_top5.result())*100))




    #values = np.arange(4*313)
    #values = values.reshape(2, 2, -1)
    #ab = tf.constant(values, dtype = tf.float64)
    #a, b = decode(ab, map_class_2_ab, 0.38)
    a = 1




