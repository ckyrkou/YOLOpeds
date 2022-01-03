#                    __
#                  / *_)
#       _.----. _ /../
#     /............./
# __/..(...|.(...|
# /__.-|_|--|_|
#
# Christos Kyrkou, PhD
# 2019

import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import cv2
from numpy import random
from .utils import parse_annotation,scale_img_anns,flip_annotations,make_target_anns,aug_img,get_boxes

import sys
sys.path.append("..")

from gen_utils import remExt

from imgaug import augmenters as iaa
from .models import custom_preprocess

CHANNEL=3

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
always = lambda aug: iaa.Sometimes(1, aug)
upto1 = lambda aug: iaa.SomeOf((1,1),aug)
upto2 = lambda aug: iaa.SomeOf((1,2),aug)
upto3 = lambda aug: iaa.SomeOf((1,3),aug)

seq = iaa.Sequential([
    always(

        [
            upto1([
                iaa.Dropout((0.01, 0.15), per_channel=0.5), # randomly remove up to 10% of the pixels
                iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=True),
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ]),
                #iaa.AverageBlur(k=(2, 3)), # blur image using local means with kernel sizes between 2 and 7
                #iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 2 and 7
                iaa.contrast.LinearContrast((0.5, 1.5), per_channel=0.5),
                #iaa.AddToHueAndSaturation((-20,20)),
                #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                #iaa.Add((-30, 30), per_channel=0.5),  # change brightness of images (by -10 to 10 of original value),
                iaa.Multiply((0.8, 1.2), per_channel=0.5),
                iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.75, 1.25))
                #iaa.UniformColorQuantization(n_colors = 64)
               ]),
        ]
    )], random_order=True) # apply augmenters in random order



def data_gen(img_folder, ann_folder, params, batch_size,preprocessingMethod,im_list=None,aug = False,has_cls=False):

    c = 0
    if(im_list == None):
        n = os.listdir(img_folder)  # List of training images
    else:
        n = im_list

    if(len(n)> 1 and aug == True):
        random.shuffle(n)

    WIDTH = params.NORM_W
    HEIGHT = params.NORM_H

    img = np.zeros((batch_size, WIDTH, HEIGHT, CHANNEL)).astype('float32')
    ann = np.zeros((batch_size, params.GRID_H, params.GRID_W, params.BOX, 4 + 1 + params.CLASS))
    cls_vec = np.zeros((batch_size, params.CLASS), dtype=np.float32)

    while (True):
        img = img*0.
        ann = ann*0.
        cls_vec = cls_vec*0.

        for i in range(c, c + batch_size):  # initially from 0 to 16, c = 0.
            train_img = cv2.imread(img_folder + '/' + remExt(n[i])+'.jpg')

            if(params.annformat=='pascalvoc'):
                train_ann = ann_folder + remExt(n[i]) + '.xml'
            if (params.annformat == 'OID'):
                train_ann = ann_folder + remExt(n[i]) + '.txt'

            bboxes = parse_annotation(train_ann, params)

            train_img,bboxes = scale_img_anns(train_img,bboxes,WIDTH,HEIGHT)

            train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)

            if(aug == True and len(bboxes)>0 and bboxes !=[]):

                if (np.random.random() < 0.5):
                    train_img = cv2.flip(train_img, 1)
                    bboxes = flip_annotations(bboxes,WIDTH,HEIGHT)

                al = []

                if (np.random.random() < 1.):  # rot
                    al.append('hsv')
                if (np.random.random() < 1.0 and len(bboxes) > 0):  # rot
                    al.append('rot')
                if (np.random.random() < 0.5 and len(bboxes) > 0):  # rot
                    al.append('sca')
                if (np.random.random() < 1.0 and len(bboxes) > 0):  # rot
                    al.append('tra')
                if (np.random.random() < 0.1 and len(bboxes) > 0):  # rot
                    al.append('she')
                if (np.random.random() < 0.5 and len(bboxes) > 0):  # rot
                    al.append('fli')

                if (len(al) > 0):
                    train_img, bboxes = aug_img(train_img, bboxes, params, al=al)

                #if (np.random.random() < 0.25):#Negative
                #    train_img = 255-train_img

                if(np.random.random() < 0.5):
                    train_img = seq.augment_images([train_img])[0]
                # else:
                if (np.random.random()<0.0 and len(bboxes)>0):
                    train_img, bboxes = aug_img(train_img,bboxes,params)

                if (np.random.random() < 0.5 and len(n)>1):
                    i2 = np.random.randint(0, len(n) - 1)
                    x2 = cv2.imread(img_folder + '/' + remExt(n[i2]) + '.jpg')
                    x2 = cv2.resize(x2, (WIDTH, HEIGHT))
                    x2 = cv2.cvtColor(x2, cv2.COLOR_BGR2RGB)

                    train_img = cv2.addWeighted(train_img, .9, x2, 0.1, 0)


            y,v = make_target_anns(bboxes,params,train_img)
            v = np.float32(v)

            train_img = train_img.astype(np.float32)

            if(preprocessingMethod == None):
                train_img = custom_preprocess(train_img)
            else:
                train_img = preprocessingMethod(train_img)

            img[i - c] = train_img

            cls_vec[i - c] = np.float32(v)
            ann[i - c] = y

        c += batch_size
        if (c + batch_size >= len(n)):
            c = 0
            if(aug == True and len(n) > 1):
                random.shuffle(n)

        if(has_cls):
            yield img, [ann,cls_vec]
        else:
            yield img, ann
