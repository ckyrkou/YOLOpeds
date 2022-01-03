import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.callbacks import Callback
from .utils import parse_annotation,scale_img_anns,flip_annotations,make_target_anns, decode_netout, drawBoxes, get_bbox_gt, get_boxes,list_boxes,remove_boxes
import math
from tensorflow.keras.models import save_model
from mean_average_precision.detection_map import DetectionMAP

from tqdm import tqdm

import sys
sys.path.append("..")

from gen_utils import remExt, hor_con, save_prev_metrics
from .models import custom_preprocess

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import datetime


def plot_loss(name,epoch,losses):
    fig = plt.figure()
    plt.plot(losses)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['loss','val_loss'])
    plt.grid()
    fig.savefig('./det_output/training_loss_'+name+'.png')
    plt.close()
    return

def plot_map(name,epoch,metrics):
    fig = plt.figure()
    plt.plot(metrics)
    plt.title('Model mAP')
    plt.ylabel('mAP')
    plt.xlabel('Epoch')
    plt.legend(['map'])
    plt.grid()
    fig.savefig('./det_output/val_map_'+name+'.png')
    plt.close()
    return

class det_callback(Callback):

    def on_train_begin(self, logs={}):
        for layer in self.model.layers:
            if (layer.name == 'class_branch'):
                self.has_cls = True
        return

    def __init__(self,num_batches,im_list,file_paths,params,preprocessingMethod,model_name,prev_metrics=[math.inf,math.inf],vis=1):
        self.im_list = im_list
        self.yolo_params = params
        self.preprocessingMethod = preprocessingMethod
        self.num_batches = num_batches
        self.losses = []
        self.metrics = []
        self.plt_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        self.loss_metrics = prev_metrics
        self.model_name = model_name
        self.best_epoch = 0
        self.im_path = file_paths[0]
        self.ann_path = file_paths[1]
        self.has_cls = False
        self.vis = vis
        self.map = 0.
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self,epoch, logs={}):
        print('\t Best Epoch: ', self.best_epoch)
        self.pbar = tqdm(total=self.num_batches+1)
        return

    def on_epoch_end(self, epoch, logs={}):

        self.losses.append([logs['loss'],logs['val_loss']])


        if(np.mod(epoch+1,100)==0):
            save_model(self.model, './saved_models/' + self.model_name + '_' + str(epoch+1) + '_.h5')
            self.model.save_weights('./saved_models/' + self.model_name + '_' + str(epoch+1) + '_weights.h5')
            print('\t -> Saving Checkpoint...')

        plot_loss(self.plt_name+'_'+self.model_name,epoch,self.losses)
        self.pbar.close()


        frames=[]
        for i in range(len(self.im_list)):

            name = remExt(self.im_list[i])
            WIDTH = self.yolo_params.NORM_W
            HEIGHT = self.yolo_params.NORM_H
            img_in = cv2.imread(self.im_path + name + '.jpg')

            if (self.yolo_params.annformat == 'pascalvoc'):
                train_ann = self.ann_path + name + '.xml'
            if (self.yolo_params.annformat == 'OID'):
                train_ann = self.ann_path + name + '.txt'

            bboxes = parse_annotation(train_ann, self.yolo_params)

            img_in, bboxes = scale_img_anns(img_in, bboxes, WIDTH, HEIGHT)

            img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)

            img = img_in.astype(np.float32)

            if (self.preprocessingMethod == None):
                img = custom_preprocess(img)
            else:
                img = self.preprocessingMethod(img)

            img = np.expand_dims(img, 0)

            net_out = self.model.predict(img, batch_size=1)
            pred = net_out.squeeze()
            image, boxes = decode_netout(img_in.copy(), pred, self.yolo_params, False, False, t_c=0.1, nms_thresh=0.5)

            b = []
            sc = []
            l = []
            idxs = []
            for box in boxes:
                b.append([box.xmin, box.ymin, box.xmax, box.ymax])
                sc.append(box.get_score())
                l.append(box.get_label())

            do_nms=False

            if (len(boxes) > 1 and do_nms==True):
                idxs = cv2.dnn.NMSBoxes(b, np.array(sc, dtype=np.float), 0.1, 0.5)
            else:
                idxs=[]
            if len(idxs) > 1:
            # loop over the indexes we are keeping
                boxes = remove_boxes(boxes, idxs)

            if(bboxes!=[]):
                gt_boxesx1y1x2y2 = np.array(bboxes[:, :4], dtype=np.float32)
                gt_labels = np.array(bboxes[:, 4], dtype=np.float32)
            else:
                gt_boxesx1y1x2y2 = np.array([], dtype=np.float32)
                gt_labels = np.array([], dtype=np.float32)

            if (boxes == []):
                bb = np.array([])
                sc = np.array([])
                l = np.array([])
                pred_boxesx1y1x2y2 = np.array([])
            else:
                bb = list_boxes(boxes, self.yolo_params)
                l = np.array(bb, dtype=np.float32)[:, 4]
                sc = np.array(bb, dtype=np.float32)[:, 5]
                pred_boxesx1y1x2y2 = np.array(bb, dtype=np.float32)[:, :4]

            frames.append((pred_boxesx1y1x2y2 / float(img_in.shape[0]),
                           l,
                           sc,
                           gt_boxesx1y1x2y2 / float(img_in.shape[1]),
                           gt_labels))

        mAP = DetectionMAP(self.yolo_params.CLASS)
        for i, frame in enumerate(frames):
            mAP.evaluate(*frame)
        map,_,_,_ = mAP.compute_mAP()

        self.metrics.append([map])

        if(map > self.map):
            print('\t e',epoch,': mAP  improved from ', self.map,' to ',map)
            save_model(self.model,'./saved_models/'+self.model_name+ '.h5')
            self.model.save_weights('./saved_models/'+self.model_name+ '_weights.h5')
            self.best_epoch = epoch + 1
            print('\t -> Saving Best...')
            self.map = map
        else:
            print('\t ',map,' mAP did not improve from ', self.map)

        plot_map(self.plt_name + '_' + self.model_name, epoch, self.metrics)


        #Get random image and do a prediction

        i = np.random.randint(0,len(self.im_list),1)[0]

        name = remExt(self.im_list[i])
        #print(i,self.im_list,name)

        WIDTH = self.yolo_params.NORM_W
        HEIGHT = self.yolo_params.NORM_H

        img_in = cv2.imread(self.im_path+name+'.jpg')

        if (self.yolo_params.annformat == 'pascalvoc'):
            train_ann = self.ann_path+name+'.xml'
        if (self.yolo_params.annformat == 'OID'):
            train_ann =self.ann_path+name+'.txt'

        bboxes = parse_annotation(train_ann, self.yolo_params)

        img_in, bboxes = scale_img_anns(img_in, bboxes, WIDTH, HEIGHT)
        #bboxes = [[16, 215, 28, 271, 2], [56, 215, 68, 271, 5]]

        img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)

        img = img_in.astype(np.float32)

        if(self.preprocessingMethod == None):
            img = custom_preprocess(img)
        else:
            img = self.preprocessingMethod(img)

        img = np.expand_dims(img, 0)

        net_out = self.model.predict(img, batch_size=1)

        if(self.has_cls):
            pred = net_out[0].squeeze()
            cls = net_out[1].squeeze()
            print(cls)
            inds = np.argsort(cls)
            for i in range(len(cls)):
                if(cls[inds[i]]>0.5):
                    print(self.yolo_params.LABELS[inds[i]])
        else:
            pred = net_out.squeeze()

        image, boxes = decode_netout(img_in.copy(), pred, self.yolo_params, False, False)

        drawBoxes(image, boxes, self.yolo_params)

        image = hor_con([img_in,image])
        cv2.putText(img=image, text="Epoch: "+str(epoch+1), org=(20,20), fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=0.55, color=(255,255,0), thickness=1)
        cv2.putText(img=image, text=name, org=(20,40), fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=0.55, color=(255,255,0), thickness=1)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if(self.vis==1):
            cv2.imshow('Image', image)
            cv2.waitKey(33)

        cv2.imwrite('./det_output/train_output_det.jpg',image)
        if((epoch+1) == self.best_epoch):
            cv2.imwrite('./det_output/best_train_output_det.jpg', image)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        self.pbar.set_description("Loss: " + str(logs['loss']))
        self.pbar.update(1)
        return