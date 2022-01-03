
import numpy as np
import numpy as np
import os
import xml.etree.ElementTree as ET
import copy
import cv2
import time
import math
import pickle



from .data_aug.data_aug import RandomRotate,RandomScale,RandomShear,RandomTranslate, Resize, Sequence,RandomHSV,RandomHorizontalFlip
from .data_aug.bbox_util import *


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def sigmoid(x):
    "Numerically-stable sigmoid function."
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)

def sigmoid_mult(x):
    "Numerically-stable sigmoid function."

    return np.where(x>=0,1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)

    if np.min(x) < t:
        x = x / (np.min(x) * t + 1e-8)

    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)

class yolo_params():

    def __init__(self,DET_TYPE,LABELS,COLORS,NORM_H,NORM_W,GRID_H,GRID_W,BOX,CLASS,CLASS_WEIGHTS,THRESHOLD,ANCHORS,SCALE_NOOB,SCALE_OBJ,SCALE_COOR,SCALE_CLASS, TRUE_BOX_BUFFER, batch_size,
                 WARMUP_BATCHES, img_dir,ann_dir,annformat):
        self.DET_TYPE = DET_TYPE
        self.LABELS = LABELS
        self.COLORS = COLORS
        self.NORM_H = NORM_H
        self.NORM_W = NORM_W
        self.GRID_H = GRID_H
        self.GRID_W = GRID_W
        self.N_GRID_H, self.N_GRID_W = int(self.NORM_H / self.GRID_H), int(self.NORM_W / self.GRID_W)
        self.BOX = BOX
        self.CLASS = CLASS
        self.CLASS_WEIGHTS = CLASS_WEIGHTS
        self.FILTER = (self.CLASS + 5) * self.BOX
        self.THRESHOLD = THRESHOLD
        self.ANCHORS = ANCHORS
        self.NOOBJ_CONF = SCALE_NOOB
        self.OBJ_CONF = SCALE_OBJ
        self.OBJ_COOR = SCALE_COOR
        self.OBJ_CLASS = SCALE_CLASS
        self.TRUE_BOX_BUFFER = TRUE_BOX_BUFFER
        self.BATCH_SIZE = batch_size
        self.WARMUP_BATCHES = WARMUP_BATCHES
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.annformat = annformat

        if(len(ANCHORS)/2 != BOX):
            error('Anchor boxes length <ANCHOR> is not equal to number of boxes defined in <BOX>')

    def print_params(self):

        print('DET TYPE',self.DET_TYPE)
        print('LABELS',self.LABELS)
        print('COLORS',self.COLORS)
        print('NORM_H',self.NORM_H)
        print('NORM_W',self.NORM_W)
        print('GRID_H',self.GRID_H)
        print('GRID_W',self.GRID_W)
        print('N_GRID_H',self.N_GRID_H)
        print('N_GRID_W',self.N_GRID_W)
        print('BOX',self.BOX)
        print('CLASS',self.CLASS)
        print('CLASS_WEIGHTS',self.CLASS_WEIGHTS)
        print('FILTER',self.FILTER)
        print('THRESHOLD',self.THRESHOLD)
        print('ANCHORS',self.ANCHORS)
        print('OBJ_CONF',self.OBJ_CONF)
        print('NOOBJ_CONF',self.NOOBJ_CONF)
        print('OBJ_COOR',self.OBJ_COOR)
        print('OBJ_CLASS',self.OBJ_CLASS)
        print('TRUE_BOX_BUFFER',self.TRUE_BOX_BUFFER)
        print('BATCH_SIZE',self.BATCH_SIZE)
        print('img_dir',self.img_dir)
        print('ann_dir',self.ann_dir)
        print('annformat',self.annformat)


def save_params(name,params):
    m = {}
    m["yolo_params"] = params.__dict__

    file = open("./saved_models/" + name + "_yolo_params.pkl", "wb")
    pickle.dump(m, file)
    file.close()

def load_params(name):
    return pickle.load( open( "./saved_models/" + name + "_yolo_params.pkl", "rb" ) )

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):

        if self.label == -1:
            if (len(self.classes) > 1):
                self.label = np.argmax(self.classes)
            else:
                self.label = 0

        return self.label

    def get_score(self):
        if self.score == -1:
            if (len(self.classes) > 1):
                self.score = self.classes[self.get_label()]
            else:
                self.score = self.c

        return self.score

    def box(self):
        return [self.xmin,self.ymin,self.xmax,self.ymax]


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / (union+1e-8)

def scale_img_anns(train_img,bboxes,WIDTH,HEIGHT):

    h, w, c = train_img.shape

    train_img = cv2.resize(train_img, (WIDTH, HEIGHT))

    for box in bboxes:
        box[0] = int(box[0] * float(WIDTH)/ w)
        box[1] = int(box[1] * float(HEIGHT)/ h)
        box[2] = int(box[2] * float(WIDTH)/ w)
        box[3] = int(box[3] * float(HEIGHT)/ h)

    return train_img,bboxes

def flip_annotations(boxes,WIDTH,HEIGHT):

    # fix object's position and size
    for box in boxes:
        xmin = box[0]
        box[0] = WIDTH - box[2]
        box[2] = WIDTH - xmin

    return boxes

def make_target_anns(boxes,params,img=None):
    ann = np.zeros((params.GRID_H, params.GRID_W, params.BOX, 4 + 1 + params.CLASS),dtype=np.float32)

    b_occ = np.zeros((params.GRID_H, params.GRID_W, params.BOX))

    anchors = [BoundBox(0, 0, params.ANCHORS[2 * i]*params.NORM_W, params.ANCHORS[2 * i + 1]*params.NORM_H) for i in range(int(len(params.ANCHORS) // 2))]

    obj_types = np.zeros((params.CLASS), dtype=np.uint32)

    epsilon = 0.1

    debug = False

    if (params.DET_TYPE == 'SSD'):
        ann[:, :, :, 4:] = [epsilon / (params.CLASS + 1)] * (params.CLASS + 1)
        ann[:, :, :, 4] = 1.-epsilon

    for box in boxes:

        center_x = .5 * (box[0] + box[2])  # xmin, xmax
        center_y = .5 * (box[1] + box[3])  # ymin, ymax


        grid_x = int(np.floor(np.mod(center_x/params.N_GRID_W,params.NORM_W/params.N_GRID_W)))
        grid_y = int(np.floor(np.mod(center_y/params.N_GRID_H,params.NORM_H/params.N_GRID_H)))

        obj_indx = int(box[4])

        center_w = (box[2] - box[0]) / (float(params.NORM_W) / params.GRID_W)  # unit: grid cell
        center_h = (box[3] - box[1]) / (float(params.NORM_H) / params.GRID_H)  # unit: grid cell

        # find the anchor that best predicts this box
        best_anchor = -1
        max_iou = -1

        obj_types[obj_indx] = 1

        b_x = 0. + (float((.5 * (box[0] + box[2]))-(grid_x*params.N_GRID_W))/float(params.N_GRID_W))
        b_y = 0. + (float((.5 * (box[1] + box[3]))-(grid_y*params.N_GRID_H))/float(params.N_GRID_H))
        b_w = (float(box[2] - box[0]))/params.NORM_W
        b_h = (float(box[3] - box[1]))/params.NORM_H

        b_t = [b_x, b_y, b_w, b_h]

        #Based on scale of window Height
        shifted_box = BoundBox(0,
                               0,
                               box[2] - box[0],
                               box[3] - box[1])

        im_box = BoundBox(0,
                               0,
                               params.NORM_W,
                               params.NORM_H)


        collision = False
        for i in range(len(anchors)):

            iouscore = bbox_iou(shifted_box,im_box)

            if(math.sqrt(iouscore) <= float((i+1))/float(len(anchors)) or collision == True):

                if b_occ[grid_y, grid_x, i] == 0:
                    best_anchor = i
                    break
                else:
                    collision=True


        if (debug):
            print('--------')
            print(b_t[0], b_t[1],b_t[3], b_t[4])
            print('Ground Truth Info: ', box, grid_x, grid_y, best_anchor, obj_indx)
            print(iouscore,math.sqrt(iouscore),best_anchor,(best_anchor+1)/len(anchors))
            print(shifted_box.xmax,shifted_box.ymax)

        if (best_anchor != -1):
            # print(best_anchor)
            if(params.DET_TYPE == 'YOLO'):
                b_occ[grid_y, grid_x, best_anchor] = 1.

                ann[grid_y, grid_x, best_anchor, 0:4] = b_t
                ann[grid_y, grid_x, best_anchor, 4] = 1.

                ann[grid_y, grid_x, best_anchor, 5:] = 0.

                ann[grid_y, grid_x, best_anchor, 5 + obj_indx] = 1. - epsilon

            if(params.DET_TYPE == 'SSD'):
                b_occ[grid_y, grid_x, best_anchor] = 1.
                ann[grid_y, grid_x, best_anchor, 0:4] = b_t
                ann[grid_y, grid_x, best_anchor, 4:] = [epsilon / (params.CLASS)] * (params.CLASS+1)
                ann[grid_y, grid_x, best_anchor, 5 + obj_indx] = 1. - epsilon


    return ann,obj_types

def xywh_to_x1y1x2y2(boxes):
    new_boxes = boxes.copy()
    new_boxes[:,0] = (boxes[:,0] - (boxes[:,2] // 2))
    new_boxes[:, 1] = (boxes[:,1] - (boxes[:,3] // 2))
    new_boxes[:, 2] = (boxes[:,0] + (boxes[:,2] // 2))
    new_boxes[:, 3] = (boxes[:,1] + (boxes[:,3] // 2))

    return new_boxes

def list_boxes(boxgt,params):

    boxes=[]

    for ind,box in enumerate(boxgt):
        max_indx = box.get_label()

        xmin_d = int(max(0, int(box.xmin)))
        ymin_d = int(max(0, int(box.ymin)))
        xmax_d = int(min(params.NORM_W, int(box.xmax)))
        ymax_d = int(min(params.NORM_H, int(box.ymax)))

        boxes.append([xmin_d,ymin_d,xmax_d,ymax_d,max_indx, box.score])

    return boxes

def get_bbox_gt(img_ann,params=None):
    img_ann = img_ann['object'][:]
    boxes = []
    obj_indx = []

    for obj in img_ann:

        if(params != None):
            obj_indx.append(params.LABELS.index(obj['name']))

        boxes.append([obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']])

    return boxes,obj_indx

def get_boxes(img_ann,params):
    bboxes = []
    obj_ann = img_ann['object'][:]

    for obj in obj_ann:
        #print(obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'])
        bboxes.append([obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], params.LABELS.index(obj['name'])])

    bboxes = np.array(bboxes, dtype=np.float32)

    return bboxes

def parse_ann_voc2(ann_dir, params=None):
    tree = ET.parse(ann_dir)
    root = tree.getroot()

    bboxes=[]

    for object in root.iter('object'):
        difficult = int(object.find('difficult').text == '1')
        # if(object.find('name').text == None):
        #     print(object.find('name').text)
        label = object.find('name').text.lower().strip()
        if label not in params.LABELS:
            continue
        else:
            label = params.LABELS.index(label)

        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        bboxes.append([xmin, ymin, xmax, ymax,label])


    bboxes = np.array(bboxes, dtype=np.float32)

    return bboxes

def parse_ann_voc(ann_dir, params=None):
    all_img = []

    tree = ET.parse(ann_dir)
    img = {'object': []}
    for elem in tree.iter():
        if 'filename' in elem.tag:
            all_img = img
            img['filename'] = elem.text
        if 'width' in elem.tag:
            img['width'] = int(elem.text)
        if 'height' in elem.tag:
            img['height'] = int(elem.text)
        if 'object' in elem.tag or 'part' in elem.tag:
            obj = {}

            for attr in list(elem):
                if 'name' in attr.tag:
                    print(attr)
                    obj['name'] = attr.text.lower()
                    if(params == None):
                        img['object'] += [obj]
                    else:
                        if obj['name'] in params.LABELS:
                            img['object'] += [obj]
                        else:
                            break

                if 'bndbox' in attr.tag:
                    for dim in list(attr):
                        if 'xmin' in dim.tag:
                            obj['xmin'] = int(round(float(dim.text)))
                        if 'ymin' in dim.tag:
                            obj['ymin'] = int(round(float(dim.text)))
                        if 'xmax' in dim.tag:
                            obj['xmax'] = int(round(float(dim.text)))
                        if 'ymax' in dim.tag:
                            obj['ymax'] = int(round(float(dim.text)))

    return all_img

def parse_ann_OID(ann_dir,params):

    classesFile = open(ann_dir, 'r')
    data = classesFile.read().strip('\n').split('\n')

    bboxes = []
    #name_of_the_class left top right bottom
    for obj in data:
        classIndex, xcen, ycen, w, h = obj.split(' ')
        classIndex = params.LABELS.index(classIndex)

        xmin = int(round(float(xcen)))
        ymin = int(round(float(ycen)))
        xmax = int(round(float(w)))
        ymax = int(round(float(h)))



        bboxes.append([xmin, ymin, xmax, ymax,classIndex])

    bboxes = np.array(bboxes, dtype=np.float32)

    return bboxes

def parse_annotation(ann_dir, params=None):

    if(params.annformat == 'pascalvoc'):

        bboxes = parse_ann_voc2(ann_dir,params)

    if(params.annformat == 'OID'):
        bboxes = parse_ann_OID(ann_dir,params)

    return bboxes


def aug_img(img,bboxes,params,al=['hsv','rot','sca','tra','she']):
    sl = []
    if ('hsv' in al):
        sl.append(RandomHSV(0.25, 0.7, 0.4))

    if ('rot' in al):
        sl.append(RandomRotate(5))
    if ('sca' in al):
        sl.append(RandomScale((-0.5, 0.5), diff=True))
    if ('tra' in al):
        sl.append(RandomTranslate(0.1, diff=True))
    if ('she' in al):
        sl.append(RandomShear(0.1))
    if ('fli' in al):
        sl.append(RandomHorizontalFlip())

    seq = Sequence(sl)

    img, bboxes = seq(img, bboxes.copy())

    #check boxes and only valid ones
    valid_boxes=[]
    for box in bboxes:
        if (box[0]<0): box[0] = 0
        if (box[1]<0): box[1] = 0
        if (box[2] >= params.NORM_W): box[2] = params.NORM_W-1
        if (box[3] >= params.NORM_H): box[3] = params.NORM_H-1

        w = box[2] - box[0]
        h = box[3] - box[1]

        if (w < 5 or h < 5):
            continue

        valid_boxes.append(box)

    return img, valid_boxes


def do_nms(boxes, nms_thresh):

    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return boxes
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0
    return boxes

def clippedrelu(x):
    return min(1.0,max(x,0.))

def decode_netout(image, netout, params, NMS=False, Draw = True, t_c=0.5,nms_thresh = 0.3,MUTE = True):


    anchors = params.ANCHORS
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []


    wh_fun = sigmoid


    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes

                if(params.DET_TYPE == 'YOLO'):
                    if(params.CLASS == 1):
                        classes = sigmoid(netout[row, col, b, 5:])
                        #print(row,col,b,classes,netout[row, col, b, 5:])
                    else:
                        #classes = _softmax(netout[row, col, b, 5:])
                        classes = sigmoid_mult(netout[row, col, b, 5:])
                else:
                    classes = _softmax(netout[row, col, b, 4:])

                # first 4 elements are x, y, w, and h
                x, y, w, h = netout[row, col, b, :4]

                if (params.DET_TYPE == 'YOLO'):
                    confidence = sigmoid(netout[row, col, b, 4])
                    thres = t_c
                    cond = (confidence * max(classes)) > thres
                else:
                    if(np.argmax(classes)>0):
                        classess = classes[1:]
                        confidence = max(classes)
                    else:
                        confidence = 0
                    cond = confidence > params.THRESHOLD


                if(cond):
                    x = int(params.N_GRID_W*(col + sigmoid(x)))  # center position, unit: image width
                    y = int(params.N_GRID_H*(row + sigmoid(y)))  # center position, unit: image height

                    factor = (params.NORM_W/params.N_GRID_W)
                    w = int(((factor*wh_fun(w)))*params.N_GRID_W) # unit: image width


                    # NORM: 0-1
                    factor = (params.NORM_H / params.N_GRID_H)
                    h = int(((factor*wh_fun(h)))*params.N_GRID_H)  # unit: image height


                    xmin = x - w / 2
                    ymin = y - h / 2
                    xmax = x + w / 2
                    ymax = y + h / 2

                    #if(b==0):
                    box = BoundBox(xmin, ymin, xmax, ymax, confidence, classes)
                    boxes.append(box)

    # suppress non-maximal boxes
    if(NMS == True):
        boxes = do_nms(boxes, nms_thresh)

    #boxes = [box for box in boxes if box.probs[np.argmax(box.probs)] > params.THRESHOLD]
    # draw the boxes using a threshold
    if(Draw == True):
        for box in boxes:
            max_indx = box.get_label()

            xmin = max(0, int(box.xmin))
            ymin = max(0, int(box.ymin))
            xmax = min(image.shape[0], int(box.xmax))
            ymax = min(image.shape[1], int(box.ymax))

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), params.COLORS[max_indx], 2)
            cv2.putText(img = image, text = params.LABELS[max_indx], org = (xmin + 10, ymin + 15), fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = (15e-4 * image.shape[0]), color = params.COLORS[max_indx], thickness = 2)

            if(MUTE == False):
                if(params.CLASS==1):
                    cv2.putText(img=image, text=str(box.c), org=(xmin + 10, ymin + 30),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=(10e-4 * image.shape[0]),color=params.COLORS[max_indx], thickness=1)
                else:
                    cv2.putText(img = image, text = str(box.get_score())+' '+str(box.c), org = (xmin + 10, ymin + 30), fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = (10e-4 * image.shape[0]), color = params.COLORS[max_indx], thickness = 1)

    return image,boxes

def drawBoxes(image, boxes, params):

    def drawbox(image,box):
        xmin = max(0, int(box.xmin))
        ymin = max(0, int(box.ymin))
        xmax = min(image.shape[0], int(box.xmax))
        ymax = min(image.shape[1], int(box.ymax))

        max_indx = box.get_label()
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), params.COLORS[max_indx], 2)
        cv2.putText(img=image, text=params.LABELS[max_indx], org=(xmin + 10, ymin + 15),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=(15e-4 * image.shape[0]),
                    color=params.COLORS[max_indx], thickness=2)

        return image

    for b in boxes:
        image = drawbox(image, b)

    return image

def remove_boxes(bboxes,indxs):
    boxes=[]

    for i in indxs.flatten():
        boxes.append(bboxes[i])

    return boxes

def remExt(n):
    if(len(n.split('.')) > 1):
        x = n.split('.')
        y = ''
        for i in range(len(x) - 2):
            y += x[i]+'.'
        y += x[-2]
        return y
    else:
        return n.split('.')[-2]

def app_det_stats(im_list,anndir,params):

    obj_hist = np.zeros((params.CLASS,))

    for i in range(len(im_list)):

        name = remExt(im_list[i])

        img_ann = parse_ann_voc(anndir + name + '.xml', params)

        bboxes = get_boxes(img_ann, params)

        for box in bboxes:
            obj_indx = int(box[4])
            obj_hist[obj_indx] += 1

    return obj_hist

