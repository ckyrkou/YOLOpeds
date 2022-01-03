import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from .utils import bbox_iou,BoundBox
import cv2

def mean_acc(params):
    def mean_acc_fixed(y_true,y_pred):
        true_xy = y_true[..., :2]
        true_wh = y_true[..., 2:4]
        true_logits = y_true[..., 5:]
		

        cell_x = tf.cast(tf.reshape(tf.tile(tf.range(params.GRID_W), [params.GRID_H]), (1, params.GRID_H, params.GRID_W, 1, 1)),dtype = tf.float32)
        cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

        cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [params.BATCH_SIZE, 1, 1, params.BOX, 1])

        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

        ### adjust w and h
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(params.ANCHORS, [1, 1, 1, params.BOX, 2])

        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])

        ### adjust class probabilities
        pred_box_class = y_pred[..., 5:]
        pred_box_prob = tf.nn.softmax(y_pred[..., 5:])

        return K.mean(tf.multiply(true_logits,pred_box_class))
    return mean_acc_fixed

def cls_acc(y_true,y_pred):

    true = tf.multiply(y_true,tf.cast(y_pred > 0.5,tf.float32))
    tot = K.sum(y_true)
    pred = K.sum(true)
    acc = tf.divide(pred,tot)
    acc = tf.reduce_mean(acc)
    #acc = K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

    return acc

def err_acc(y_true,y_pred):

    error = tf.reduce_sum(tf.abs(y_true-y_pred))

    return error


def IoU(params):
    def IoU_fixed(y_true,y_pred):
        n_cells = y_pred.get_shape().as_list()[1]

        predicted_xy_s = tf.nn.sigmoid(y_pred[..., :2])

        cell_inds = tf.range(n_cells, dtype=tf.float32)

        predicted_xy = tf.stack((
            (predicted_xy_s[..., 0] + tf.reshape(cell_inds, [1, -1, 1])) * params.N_GRID_W,
            (predicted_xy_s[..., 1] + tf.reshape(cell_inds, [-1, 1, 1])) * params.N_GRID_H
        ), axis=-1)

        predicted_wh_s = tf.sigmoid(y_pred[..., 2:4])
        predicted_wh = predicted_wh_s

        predicted_wh = tf.stack((
            predicted_wh[..., 0] * float(params.NORM_W),
            predicted_wh[..., 1] * float(params.NORM_H)
        ), axis=-1)

        predicted_min = predicted_xy - predicted_wh / 2
        predicted_max = predicted_xy + predicted_wh / 2

        true_xy_s = y_true[..., :2]
        true_wh_s = y_true[..., 2:4]

        true_xy = tf.stack((
            (true_xy_s[..., 0] + tf.reshape(cell_inds, [1, -1, 1])) * tf.constant(float(params.N_GRID_W)),
            (true_xy_s[..., 1] + tf.reshape(cell_inds, [-1, 1, 1])) * tf.constant(float(params.N_GRID_H))
        ), axis=-1)

        true_wh = tf.stack((
            true_wh_s[..., 0] * tf.constant(float(params.NORM_W)),
            true_wh_s[..., 1] * tf.constant(float(params.NORM_H))
        ), axis=-1)

        true_min = true_xy - true_wh / 2
        true_max = true_xy + true_wh / 2

        intersect_mins = tf.maximum(predicted_min, true_min)
        intersect_maxes = tf.minimum(predicted_max, true_max)
        intersect_wh = intersect_maxes - intersect_mins
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = predicted_wh[..., 0] * predicted_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.maximum(intersect_areas / (union_areas + 1e-8), 0.)

        selector = y_true[..., 4]

        ioum = tf.divide(tf.reduce_sum(tf.multiply(selector, iou_scores)), tf.reduce_sum(selector) + 1e-8)

        return ioum

    return IoU_fixed

def obj_acc(y_true,y_pred):

    selector = y_true[..., 4]
		
    predicted_objectedness = tf.cast(tf.nn.sigmoid(y_pred[..., 4]) > 0.5,dtype = tf.float32)

    acc = tf.divide(tf.reduce_sum(tf.multiply(selector,predicted_objectedness)),tf.reduce_sum(selector)+1e-8)


    return acc

def class_acc(params):
    def class_acc_fixed(y_true,y_pred):

        selector = y_true[..., 4]
        true_class = y_true[..., 5:]

        if (params.CLASS == 1):
            predicted_class = tf.nn.sigmoid(y_pred[..., 5:])
        else:
            predicted_class = tf.nn.softmax(y_pred[..., 5:])
		
		
        acc = tf.divide(tf.reduce_sum(tf.multiply(tf.cast(predicted_class>0.5,dtype = tf.float32),true_class)),tf.reduce_sum(selector)+1e-8)

        return acc

    return class_acc_fixed

def no_obj_acc(y_true,y_pred):

    negselector = 1-y_true[..., 4]
    #selector = y_true[..., 4]

    predicted_objectedness = tf.cast(tf.nn.sigmoid(y_pred[..., 4]) > 0.5,dtype = tf.float32)

    acc = tf.divide(tf.reduce_sum(tf.multiply(negselector,predicted_objectedness)),tf.reduce_sum(predicted_objectedness)+1e-8)

    return acc

def comb_metric(params):
    def comb_metric_fixed(y_true,y_pred):
        m = class_acc(params)(y_true,y_pred) + obj_acc(y_true,y_pred) + IoU(params)(y_true,y_pred) - no_obj_acc(y_true,y_pred)
        return m
    return comb_metric_fixed

def mAP_metric(params):
    def comb_metric_fixed(y_true,y_pred):
        m = class_acc(params)(y_true,y_pred)+obj_acc(y_true,y_pred)+IoU(params)(y_true,y_pred)
        return m
    return comb_metric_fixed

def eval_det_perf(all_obj,boxes,params,info=False):


    class_dict = {}
    for i in range(0, len(params.LABELS)):
        class_dict[params.LABELS[i]] = i

    total_objects = np.zeros((len(params.LABELS)))

    metrics = np.zeros((len(params.LABELS),3))

    avg_iou = np.zeros((len(params.LABELS)))

    for i in range(0,len(params.LABELS)):
        class_dict[params.LABELS[i]] = i

    #print("Total true boxes:",len(all_obj))
    for ind,obj in enumerate(all_obj):

        cls_o = int(obj[4])

        total_objects[cls_o]+=1

        xmin_o = int(obj[0])
        ymin_o = int(obj[1])
        xmax_o = int(obj[2])
        ymax_o = int(obj[3])

        iou_max = 0


        cand_i = -1

        class_i = -1


        if(len(boxes)>0):
            for i,box in enumerate(boxes):
                max_indx = box.get_label()

                xmin_d = int(max(0, int(box.xmin)))
                ymin_d = int(max(0, int(box.ymin)))
                xmax_d = int(min( params.NORM_W, int(box.xmax)))
                ymax_d = int(min(params.NORM_H, int(box.ymax)))

                iou = bbox_iou(BoundBox(xmin_d,ymin_d,xmax_d,ymax_d),BoundBox(xmin_o,ymin_o,xmax_o,ymax_o))

                if(iou>iou_max and iou >0.3):
                    iou_max = iou
                    cand_i = i
                    class_i = max_indx

            if (cand_i == -1 or class_i != cls_o):# no associated prediction
                metrics[cls_o, 2] += 1 # FN

            else:
                metrics[class_i, 0] += 1 # TP
                del boxes[cand_i]
                avg_iou[class_i] += iou_max
        else:
            metrics[cls_o, 2] += 1

    for i, box in enumerate(boxes): #remaining boxes did not have good IoU, so are false positives
        max_indx = box.get_label()
        metrics[max_indx, 1] += 1  # FP

    false_dets = np.sum(metrics[:,1]) #FP
    detected_objects = np.sum(metrics[:,0]) #TP
    missed_dets = np.sum(metrics[:,2]) #FN

    if(info == True):
        print('------------')
        print(np.sum(total_objects))

        print('Accuracy Stats')
        for i in range(len(metrics)):
            s = ' ' + params.LABELS[i] + ': '
            s = s + ' TP: ' + str(metrics[i, 0]) + ' FP:  ' + str(metrics[i, 1]) + ' FN: ' + str(metrics[i, 2])
            print(s)

        print('Average IoU per Class:')
        for i in range(len(avg_iou)):
            s = ' ' + params.LABELS[i] + ': '
            s = s + ' ' + str(avg_iou[i] / np.max((total_objects[i], 1)))
            print(s)

        print('False Detections: ', false_dets, ' Correct Detections: ', detected_objects, ' Missed Detections: ',
              missed_dets)
        print('------------')

    return metrics,avg_iou, total_objects, false_dets,detected_objects,missed_dets