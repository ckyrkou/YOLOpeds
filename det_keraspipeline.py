#                    __
#                  / *_)
#       _.----. _ /../
#     /............./
# __/..(...|.(...|
# /__.-|_|--|_|
#
# Christos Kyrkou, PhD
# 2020

import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau, LearningRateScheduler,TerminateOnNaN
from tensorflow.keras.optimizers import Adam, SGD, Adadelta
from tensorflow.keras.models import  load_model, save_model

import numpy as np
import os
import random as rnd

from gen_utils import *

from detection.utils import *
from detection.losses import yolo_loss_v2,cls_loss,ssd_loss
from detection.metrics import mean_acc, cls_acc, err_acc, obj_acc, IoU, class_acc, comb_metric, no_obj_acc
from detection.models import MYMODEL,get_preprocess_method, LBCconv, LBCconv1x1,CSLBCconv,HaarCconv,LBPact,Mish
from detection.app_params import *
from detection.callbacks import det_callback
from detection.datagen import data_gen
from detection.schedulers import cosine_decay,lr_schedule_fix,lr_schedule_cycle, cosine_annealing,lr_no_schedule,lr_schedule_step

from tensorflow.python.framework.ops import disable_eager_execution

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-m','-model', dest='model',required=True,
                    help='Select backbone model from: [vgg,resnet,mobileV2,mobile,ACFF14,ACFF12,ACFF,miniGoogleNet,yolopeds,vanillaConv,squeezenet,LBP,dronet,tinyyolov2, CSLBP]')

parser.add_argument('-d','-dataset', dest='dataset',required=True,
                    help='Select dataset to use from: [voc_det_seg,kangaroo,imb,kanga2,pennfudan,pets_2009,raccoon,air_cars_big,air_cars,voc_2012_det,PennFudanPed]')

parser.add_argument('-r','-resume', dest='resumeTraining',required=False,default=0,type=int,
                    help='Continue training form a previous checkpoint for this dataset.')

parser.add_argument('-e','-epochs', dest='epochs',required=False,default=400,type=int,
                    help='Number of epochs for training.')

parser.add_argument('-l','-lr', dest='learningrate',required=False,default=1e-3,type=float,
                    help='Initial learning rate.')

parser.add_argument('-o','-optimizer', dest='optimizer',required=False,default='sgd',
                    help='Select optimizer from [adam,sgd].')

parser.add_argument('-s','-scheduler', dest='scheduler',required=False,default='none',
                    help='Select scheduler from [none, cosine, fix (reduce by 10 every 100 epochs)].')

parser.add_argument('-w','-weights', dest='classweights',required=False,default=0,type=int,
                    help='Calculate weights for each class depending on the instances per class')

parser.add_argument('-z','-visualize', dest='visualize',required=False,default=1,type=int,
                    help='Visualize training images and detections every epoch.')

inargs = parser.parse_args()
print(inargs)

disable_eager_execution()

os.environ["CUDA_VISIBLE_DEVICES"]="0"


app = inargs.dataset

task = 'det'

#0 - start fresh, 1 - load only model, 2 - also load metrics
resumeTraining = inargs.resumeTraining

NO_OF_EPOCHS = inargs.epochs

root = './data/'+app

train_frame_path = root + '/train_im/'
train_mask_path = root + '/train_mask/'
train_ann_path = root + '/train_ann/'

val_frame_path = root + '/val_im/'
val_mask_path = root + '/val_mask/'
val_ann_path = root + '/val_ann/'

im_list = os.listdir(train_frame_path)
val_list = os.listdir(val_frame_path)

NO_OF_TRAINING_IMAGES = len(im_list)
NO_OF_VAL_IMAGES = len(val_list)

BATCH_SIZE = min(32,NO_OF_TRAINING_IMAGES,NO_OF_VAL_IMAGES)
lr = inargs.learningrate

print('Total Iterations:',NO_OF_EPOCHS*(NO_OF_TRAINING_IMAGES//BATCH_SIZE))

print(NO_OF_TRAINING_IMAGES,NO_OF_VAL_IMAGES,BATCH_SIZE)

params = get_app_params(app, BATCH_SIZE)
params.img_dir =train_frame_path
params.ann_dir =train_ann_path

WIDTH = params.NORM_W
HEIGHT = params.NORM_H

if(inargs.classweights==1):
    obj_count = app_det_stats(im_list,train_ann_path,params)
    weights = np.array(np.min(obj_count) / obj_count, dtype=np.float32)
    weights[weights<0.1]=0.1
    print('CLASS WEIGHTS')
    print(weights)
else:
    weights = None


if(params.DET_TYPE == 'YOLO'):
    loss = yolo_loss_v2(params,weights,debug = False,iou_thresh = 0.6,class_loss = 'MSE',scale=1.0)
else:
    loss = ssd_loss(params)


metric = mean_acc(params)
if(inargs.optimizer =='adam'):
    opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
if (inargs.optimizer == 'sgd'):
    opt = SGD(lr=lr, momentum=0.9, nesterov=True,clipvalue=2000.0)

earlystopping = EarlyStopping(monitor = 'METRIC_TO_MONITOR', verbose = 1,
                              min_delta = 0.01, patience = 3, mode = 'max')

lrr = ReduceLROnPlateau(factor=0.75, patience=10,verbose=1,min_lr=1e-6)

scheduler = lr_no_schedule
if(inargs.scheduler == 'cosine'):
    scheduler = cosine_decay(epochs_tot=NO_OF_EPOCHS,initial_lrate=lr)
if(inargs.scheduler == 'fix'):
    scheduler = lr_schedule_fix
if(inargs.scheduler == 'step'):
    scheduler = lr_schedule_step

lrs = LearningRateScheduler(scheduler,verbose=1)

seed = 11
rnd.seed(seed)
np.random.seed(seed)

net_name=inargs.model

model_name = net_name + '_' + task + '_' + app

preprocessing = get_preprocess_method(net_name)

if(resumeTraining == 0):
    cnn = MYMODEL(params,net_name)
    prev_metrics = [math.inf, math.inf]
else:

    cnn = load_model('./saved_models/'+model_name+'.h5',custom_objects={'yolo_loss_fixed':yolo_loss_v2(params),'mean_acc_fixed':mean_acc(params),'cls_loss':cls_loss,'cls_acc':cls_acc,'no_obj_acc':no_obj_acc,'obj_acc':obj_acc,'IoU':IoU(params),
                                         'class_acc':class_acc(params),'comb_metric':comb_metric(params),'IoU_fixed':IoU(params),'class_acc_fixed':class_acc(params),'comb_metric_fixed':comb_metric(params),
                                                                      'Mish':Mish,'LBCconv':LBCconv,'LBCconv1x1':LBCconv1x1,'CSLBCconv':CSLBCconv,'HaarCconv':HaarCconv,'LBPact':LBPact})
    if (resumeTraining == 2):
        prev_metrics = load_prev_metrics(model_name)
        print('... Resumed training for model ',model_name,' Last Metrics: ',prev_metrics)
    else:
        prev_metrics = [math.inf, math.inf]
        print('... Resumed training for model ', model_name, ' No Metrics loaded')

monit = det_callback(NO_OF_TRAINING_IMAGES // BATCH_SIZE,val_list, [val_frame_path, val_ann_path], params, preprocessing, model_name, prev_metrics,vis=inargs.visualize)
tnan = TerminateOnNaN()

cnn.summary()
lq.models.summary(cnn)
net_flops(cnn, table=True)


for layer in cnn.layers:
    if(layer.name == 'class_branch'):
        print('[INFO] Has auxiliary classification layer')
        has_cls = True

        #weights = np.array(1-obj_count/np.sum(obj_count), dtype=np.float32)
        weights = np.array(np.max(obj_count)/obj_count, dtype=np.float32)
        print(np.shape(weights))
        print(weights)

        loss_mat = {'yolo_head':loss,'class_branch':cls_loss(weights)}
        metrics_mat = {'class_branch':cls_acc,'yolo_head':[IoU(params),obj_acc,class_acc(params),comb_metric(params),no_obj_acc]}
    else:
        loss_mat = {'yolo_head':loss}
        metrics_mat = {'yolo_head':[IoU(params),obj_acc,class_acc(params),comb_metric(params),no_obj_acc]}
        has_cls = False

# Train the model
train_gen = data_gen(train_frame_path, train_ann_path, params, preprocessingMethod=preprocessing,batch_size=BATCH_SIZE,im_list=im_list,aug = True,has_cls=has_cls)
val_gen = data_gen(val_frame_path, val_ann_path, params, preprocessingMethod=preprocessing,batch_size=BATCH_SIZE,im_list=val_list,aug = False,has_cls=has_cls)

save_params(model_name,params)

cnn.compile(loss=loss_mat,
              optimizer=opt,
              metrics=metrics_mat)

checkpoint = ModelCheckpoint('./saved_models/'+model_name+ '_metric.h5', monitor='val_comb_metric_fixed', save_best_only=True, mode='max', verbose=1, save_weights_only=False)
weight_checkpoint = ModelCheckpoint('./saved_models/'+model_name+ '_metric_weights.h5', monitor='val_comb_metric_fixed', save_best_only=True, mode='max', verbose=1, save_weights_only=True)

callbacks_list = [lrs,monit,tnan,checkpoint,weight_checkpoint]

results = cnn.fit_generator(generator=train_gen, validation_data=val_gen,epochs=NO_OF_EPOCHS,
                          steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                          validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE),
                          callbacks=callbacks_list,verbose=2)

save_model(cnn,'./saved_models/'+model_name+'_last.h5')
print('\t -> Saving Last...')