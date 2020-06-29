

import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D, Reshape, Multiply, MaxPooling2D, Cropping2D, \
    UpSampling2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras import Model
from LoadDataset2 import *
import cv2
import glob
import pickle
import numpy as np

files = "./test2014/*.jpg"
imges = [f for f in glob.glob(files)]

def create_dictionaries2(col_int, int_col, images):
    N = 313
    f_shape = (256, 256)
    space = np.zeros((25, 25))
    for i in range(len(images)):
        height, width, _  = images[i].shape
        img = cv2.resize(images[i], f_shape, interpolation = cv2.INTER_AREA)
        lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        for x in range(f_shape[0]):
            for y in range(f_shape[1]):
                a, b = lab_image[x, y][1:]
                if a < 0 or b < 0:
                    print(a, b)
                space[a // 10, b // 10] = space[a // 10, b // 10] + 1

    space_1d = space.flatten()
    idx_1d = space_1d.argsort()[-N:]
    x_idx, y_idx = np.unravel_index(idx_1d, space.shape)

    point_list = np.zeros((N, 2))

    for i in range(x_idx.shape[0]):
        tup = [x_idx[i]*10, y_idx[i]*10]
        point_list[i, 0] = tup[0]
        point_list[i, 1] = tup[1]
        col_int[str(tup)] = i
        int_col[str(i)] = tup

    return col_int, int_col, point_list





def create_dictionaries(col_int, int_col, images):
    i = 0
    for img in images:
        height, width, _  = img.shape
        lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        for x in range(height):
            for y in range(width):
                key = str([lab_image[x, y][1:]])
                if key not in col_int.keys():
                    col_int[key] = i
                    int_col[str(i)] = lab_image[x, y][1:]
                    i = i + 1

    return col_int, int_col


def datas():
    dim = (256, 256)

    train_images = []
    for files in imges:
        if len(train_images) == 0:
            train_images = [cv2.imread(files)]
        else:
            train_images.append(cv2.imread(files))

    ab_2_class = {}
    class_2_ab = {}
    ab_2_class, class_2_ab, ab_vals = create_dictionaries2(ab_2_class, class_2_ab, train_images[:1000])
    Dicts = [ab_2_class, class_2_ab, ab_vals]

    with open('dictionaries.p', 'wb') as fp:
        pickle.dump(Dicts, fp, protocol = pickle.HIGHEST_PROTOCOL)



    img = train_images[0]
    reshaped_image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    print(reshaped_image.shape)
    cv2.imshow("Imagen", reshaped_image)
    cv2.waitKey(0)



if __name__ == "__main__":
    a = (1, 2)
    c = str(a)
    datas()
