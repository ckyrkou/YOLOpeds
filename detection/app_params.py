#                    __
#                  / *_)
#       _.----. _ /../
#     /............./
# __/..(...|.(...|
# /__.-|_|--|_|
#
# Christos Kyrkou, PhD
# 2019


from .utils import yolo_params
import numpy as np
import pickle


def get_app_params(app, batch_size):

    annformat = 'pascalvoc'

    if (app == 'pets_2009'):
        COLORS = [(128, 128, 0)]

        LABELS = ['ped']

        NORM_H, NORM_W = 320, 320
        GRID_H, GRID_W = 10, 10
        N_GRID_H, N_GRID_W = int(NORM_H / GRID_H), int(NORM_W / GRID_W)
        ANCHORS = np.array([1.05,1.86, 3.14,4.51, 4.18,9.73, 8.51,9.36, 13.49,13.02])*24/320  # VOC
        BOX = int(len(ANCHORS) / 2)
        CLASS = len(LABELS)
        CLASS_WEIGHTS = np.ones(CLASS, dtype='float32')
        FILTER_YOLO = (CLASS + 5) * BOX
        FILTER_SSD = (CLASS + 1+4) * BOX
        THRESHOLD = 0.5
        #NOOBJ_CONF, OBJ_CONF, OBJ_COOR, OBJ_CLASS = 0.0, 0.0, 0.0, 0.0
        NOOBJ_CONF, OBJ_CONF, OBJ_COOR, OBJ_CLASS = 1., 1.0, 1.0, 1.0
        #NOOBJ_CONF, OBJ_CONF, OBJ_COOR, OBJ_CLASS = 0.05, 2.0, 1.0, 2.0
        TRUE_BOX_BUFFER = 100
        WARMUP_BATCHES = 0
        DET_TYPE = 'YOLO'
        ann_dir = './data/JPG_VOC/'
        #ann_dir = './Annotations/JPG_VOC/train/'
        #val_ann_dir = './Annotations/JPG_VOC/val/'
        img_dir = './dataset/JPG_VOC/'
        #img_dir = './dataset/JPG_VOC/train/'
        #val_img_dir = './dataset/JPG_VOC/val/'


    if (app == 'pennfudan'):
        COLORS = [(128, 128, 0)]

        LABELS = ['ped']

        NORM_H, NORM_W = 320, 320
        GRID_H, GRID_W = 10, 10
        N_GRID_H, N_GRID_W = int(NORM_H / GRID_H), int(NORM_W / GRID_W)
        ANCHORS = np.array([1.05,1.86, 3.14,4.51, 4.18,9.73, 8.51,9.36, 13.49,13.02])*24/320  # VOC
        BOX = int(len(ANCHORS) / 2)
        CLASS = len(LABELS)
        CLASS_WEIGHTS = np.ones(CLASS, dtype='float32')
        FILTER_YOLO = (CLASS + 5) * BOX
        FILTER_SSD = (CLASS + 1+4) * BOX
        THRESHOLD = 0.5
        #NOOBJ_CONF, OBJ_CONF, OBJ_COOR, OBJ_CLASS = 0.0, 0.0, 0.0, 0.0
        NOOBJ_CONF, OBJ_CONF, OBJ_COOR, OBJ_CLASS = 0.1, 1.0, 5.0, 1.0
        #NOOBJ_CONF, OBJ_CONF, OBJ_COOR, OBJ_CLASS = 0.05, 2.0, 1.0, 2.0
        TRUE_BOX_BUFFER = 100
        WARMUP_BATCHES = 0
        DET_TYPE = 'YOLO'
        ann_dir = './data/JPG_VOC/'
        #ann_dir = './Annotations/JPG_VOC/train/'
        #val_ann_dir = './Annotations/JPG_VOC/val/'
        img_dir = './dataset/JPG_VOC/'
        #img_dir = './dataset/JPG_VOC/train/'
        #val_img_dir = './dataset/JPG_VOC/val/'

    if (app == 'voc_2012_det'):
        COLORS = [(128, 0, 0), (0, 128, 0), (128, 128, 0),
                        (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
                        (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                        (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                        (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                        (0, 64, 128)]

        LABELS = ['aeroplane', 'bicycle', 'bird', 'boat',
                      'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                      'diningtable', 'dog', 'horse', 'motorbike', 'person',
                      'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        NORM_H, NORM_W = 416, 416
        GRID_H, GRID_W = 13, 13
        N_GRID_H, N_GRID_W = int(NORM_H / GRID_H), int(NORM_W / GRID_W)
        ANCHORS = np.array([1.05,1.86, 3.14,4.51, 4.18,9.73, 8.51,9.36, 13.49,13.02])*(22/320)  # VOC
        BOX = int(len(ANCHORS) / 2)
        BOX = int(len(ANCHORS) / 2)
        CLASS = len(LABELS)
        CLASS_WEIGHTS = np.ones(CLASS, dtype='float32')
        FILTER_YOLO = (CLASS + 5) * BOX
        FILTER_SSD = (CLASS + 1+4) * BOX
        THRESHOLD = 0.5
        NOOBJ_CONF, OBJ_CONF, OBJ_COOR, OBJ_CLASS = 1., 1.0, 5.0, 1.0
        #NOOBJ_CONF, OBJ_CONF, OBJ_COOR, OBJ_CLASS = 0.05, 2.0, 1.0, 2.0
        TRUE_BOX_BUFFER = 100
        WARMUP_BATCHES = 0
        DET_TYPE = 'YOLO'
        ann_dir = './data/JPG_VOC/'
        #ann_dir = './Annotations/JPG_VOC/train/'
        #val_ann_dir = './Annotations/JPG_VOC/val/'
        img_dir = './dataset/JPG_VOC/'
        #img_dir = './dataset/JPG_VOC/train/'
        #val_img_dir = './dataset/JPG_VOC/val/'


    if (app == 'vedai'):
        COLORS = [(128, 0, 0), (0, 128, 0), (128, 128, 0),
                        (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
                        (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                        (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                        (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                        (0, 64, 128)]

        LABELS = ['car']

        NORM_H, NORM_W = 512, 512
        GRID_H, GRID_W = 16, 16
        N_GRID_H, N_GRID_W = int(NORM_H / GRID_H), int(NORM_W / GRID_W)
        ANCHORS = np.array([1.05,1.86, 3.14,4.51, 4.18,9.73, 8.51,9.36, 13.49,13.02])*(22/320)  # VOC
        BOX = int(len(ANCHORS) / 2)
        CLASS = len(LABELS)
        CLASS_WEIGHTS = np.ones(CLASS, dtype='float32')
        FILTER_YOLO = (CLASS + 5) * BOX
        FILTER_SSD = (CLASS + 1+4) * BOX
        THRESHOLD = 0.5
        NOOBJ_CONF, OBJ_CONF, OBJ_COOR, OBJ_CLASS = 0.5, 1.0, 5.0, 1.0
        #NOOBJ_CONF, OBJ_CONF, OBJ_COOR, OBJ_CLASS = 0.05, 2.0, 1.0, 2.0
        TRUE_BOX_BUFFER = 100
        WARMUP_BATCHES = 0
        DET_TYPE = 'YOLO'
        ann_dir = './data/JPG_VOC/'
        #ann_dir = './Annotations/JPG_VOC/train/'
        #val_ann_dir = './Annotations/JPG_VOC/val/'
        img_dir = './dataset/JPG_VOC/'
        #img_dir = './dataset/JPG_VOC/train/'
        #val_img_dir = './dataset/JPG_VOC/val/'


    if (app == 'kangaroo'):
        LABELS = ['kangaroo']
        COLORS = [(0, 0, 255)]
        NORM_H, NORM_W = 320, 320
        GRID_H, GRID_W = 10, 10
        N_GRID_H, N_GRID_W = int(NORM_H / GRID_H), int(NORM_W / GRID_W)
        ANCHORS = [5.03,5.62, 7.29,17.55, 13.27,24.55, 14.30,13.39, 23.19,22.60]
        BOX = int(len(ANCHORS) / 2)
        CLASS = len(LABELS)
        CLASS_WEIGHTS = np.ones(CLASS, dtype='float32')
        FILTER = (CLASS + 5) * BOX
        THRESHOLD = 0.5
        #NOOBJ_CONF, OBJ_CONF, OBJ_COOR, OBJ_CLASS = 0.5, 3.0, 2.0, 2.0
        NOOBJ_CONF, OBJ_CONF, OBJ_COOR, OBJ_CLASS = 0.05, 1.0, 10.0, 1.0
        TRUE_BOX_BUFFER = 10
        WARMUP_BATCHES = 0
        DET_TYPE = 'YOLO'
        # ann_dir = './dataset/JPG/'
        # img_dir = './dataset/JPG/'
        ann_dir = './dataset/JPG_Kanga/'
        img_dir = './dataset/JPG_Kanga/'

    if (app == 'facemask'):
        LABELS = ['mask','no-mask']
        COLORS = [(0, 0, 255),(255, 0, 0)]
        NORM_H, NORM_W = 320, 320
        GRID_H, GRID_W = 10, 10
        N_GRID_H, N_GRID_W = int(NORM_H / GRID_H), int(NORM_W / GRID_W)
        ANCHORS = [5.03,5.62, 7.29,17.55, 13.27,24.55, 14.30,13.39, 23.19,22.60]
        BOX = int(len(ANCHORS) / 2)
        CLASS = len(LABELS)
        CLASS_WEIGHTS = np.ones(CLASS, dtype='float32')
        FILTER = (CLASS + 5) * BOX
        THRESHOLD = 0.5
        #NOOBJ_CONF, OBJ_CONF, OBJ_COOR, OBJ_CLASS = 0.5, 3.0, 2.0, 2.0
        NOOBJ_CONF, OBJ_CONF, OBJ_COOR, OBJ_CLASS = 0.1, 1.0, 5.0, 1.0
        TRUE_BOX_BUFFER = 10
        WARMUP_BATCHES = 0
        DET_TYPE = 'YOLO'
        # ann_dir = './dataset/JPG/'
        # img_dir = './dataset/JPG/'
        ann_dir = './dataset/JPG_Kanga/'
        img_dir = './dataset/JPG_Kanga/'

    if (app == 'raccoon'):
        LABELS = ['raccoon']
        COLORS = [(0, 0, 255)]
        NORM_H, NORM_W = 320, 320
        GRID_H, GRID_W = 10, 10
        N_GRID_H, N_GRID_W = int(NORM_H / GRID_H), int(NORM_W / GRID_W)
        ANCHORS = [5.03,5.62, 7.29,17.55, 13.27,24.55, 14.30,13.39, 23.19,22.60]
        BOX = int(len(ANCHORS) / 2)
        CLASS = len(LABELS)
        CLASS_WEIGHTS = np.ones(CLASS, dtype='float32')
        FILTER = (CLASS + 5) * BOX
        THRESHOLD = 0.5
        #NOOBJ_CONF, OBJ_CONF, OBJ_COOR, OBJ_CLASS = 0.5, 3.0, 2.0, 2.0
        NOOBJ_CONF, OBJ_CONF, OBJ_COOR, OBJ_CLASS = 0.1, 1.0, 5.0, 1.0
        TRUE_BOX_BUFFER = 10
        WARMUP_BATCHES = 0
        DET_TYPE = 'YOLO'
        # ann_dir = './dataset/JPG/'
        # img_dir = './dataset/JPG/'
        ann_dir = './dataset/JPG_Kanga/'
        img_dir = './dataset/JPG_Kanga/'


    params = yolo_params(DET_TYPE, LABELS, COLORS, NORM_H, NORM_W, GRID_H, GRID_W, BOX, CLASS, CLASS_WEIGHTS, THRESHOLD,
                             ANCHORS, NOOBJ_CONF, OBJ_CONF, OBJ_COOR, OBJ_CLASS, TRUE_BOX_BUFFER, batch_size, WARMUP_BATCHES,img_dir, ann_dir, annformat)

    params.print_params()

    return params