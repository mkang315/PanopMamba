import os
import numpy as np
import glob
import cv2
import scipy.io as sio
import skimage
import torch
from PIL import Image
import scipy
import scipy.ndimage
from skimage import measure
from torchmetrics.detection import PanopticQuality
def get_all_distance(img, label_dic, dist):

    # img : input label image
    # label_dic : labels dictionary
    # dist : L1, L2, Chess, etcs
    dist_map = np.zeros([img.shape[0],img.shape[1]])


    for key_idx in label_dic.keys():

        label = np.abs(img - label_dic[key_idx])
        label_one_ch = label[:,:]

        target_label = label_one_ch == 0
        target_label = target_label.astype(np.uint8)
        dist_transform = cv2.distanceTransform(target_label, cv2.DIST_L2, 5)

        dist_map += dist_transform/(np.max(dist_transform)+0.01)


    return dist_map

def wIoU_evaluation(label_dic, gt_img, t_img, weight_map):
    # label_dic : Dictionary for class labels
    # gt_img    : Ground-truth image
    # t_img     : Predicted image
    # weighted map : weight map
    img_H = gt_img.shape[0]
    img_W = gt_img.shape[1]
    iou_stack = {}
    for key_idx in label_dic.keys():
        GT = 0
        TP = 0
        FP = 0
        IoU = 0
        gt_map   = np.zeros((img_H,img_W,1))
        test_map = np.zeros((img_H,img_W,1))

        gt_idx   = gt_img == label_dic[key_idx]
        test_idx = t_img  == label_dic[key_idx]

        gt_map[gt_idx[:,:]] = 1
        test_map[test_idx[:,:]] = 1
        for i in range(gt_map.shape[0]):
            for j in range(gt_map.shape[1]):

                if gt_map[i,j] == 1:
                    GT += weight_map[i,j]

                if gt_map[i,j] == 1 and test_map[i,j] == 1:
                    TP += weight_map[i,j]

                elif gt_map[i,j] == 0 and test_map[i,j] == 1:
                    FP += weight_map[i,j]
        if GT == 0:
            pass
            #print('no label')

        else:
            IoU = TP/(GT+FP)
            #iou_stack.append(IoU)
            #print(IoU)
        iou_stack[key_idx] = IoU
   # meaniou = sum(iou_stack)/len(iou_stack)

    return iou_stack

def Panoptic_quality(ground_truth_image,predicted_image,iou):
    TP = 0
    FP = 0
    FN = 0
    sum_IOU = 0
    matched_instances = {}
    vk = {v:k for k,v in label_dic.items()}
    #PQ_calculation(ground_truth_image,predicted_image)
    for i in np.unique(ground_truth_image):
        if i == 0:
            pass
        else:
            temp_image = np.array(ground_truth_image)
            temp_image = temp_image == i
            matched_image = temp_image * predicted_image

            for j in np.unique(matched_image):
                if j == 0:
                    pass
                else:

                    IOU =  iou[vk[i]]
                    if IOU> 0.5:
                        matched_instances [i] = j, IOU

    pred_indx_list = np.unique(predicted_image)
    pred_indx_list = np.array(pred_indx_list[1:])

    # Loop on ground truth instances
    for indx in np.unique(ground_truth_image):
        if indx == 0:
            pass
        else:
            if indx in matched_instances.keys():
                pred_indx_list = np.delete(pred_indx_list, np.argwhere(pred_indx_list == [indx][0]))
                TP = TP+1
                sum_IOU = sum_IOU+matched_instances[indx][1]
            else:
                FN = FN+1
    FP = len(np.unique(pred_indx_list))
    try:
        PQ = sum_IOU/(TP+0.5*FP+0.5*FN)
        SQ = sum_IOU/TP
        RQ = TP/(TP+0.5*FP+0.5*FN)
    except:
        PQ = 0
        SQ = 0
        RQ = 0

    return PQ,SQ,RQ

alltruth = [x for x in glob.glob(os.path.join("./txtoutputs/real/*.png"))  if x.find("_.png")<0]
alltruth = sorted(alltruth,key=lambda x: x.split(os.sep)[-1].split(".")[0])
allpred = [x for x in glob.glob(os.path.join("./txtoutputs/predict/*.png"))  if x.find("_.png")<0]
allpred = sorted(allpred,key=lambda x:  x.split(os.sep)[-1].split(".")[0])


allpqs = []
allsq = []
allrq = []
alpha  = 0.01
dist = cv2.DIST_L2
label_dic={
        'Epithelial':1,
        'Lymphocyte':2,
        'Neutrophil':3,
        'Macrophage' :4
      }
i = 0
for truth,pred in zip(alltruth,allpred):
    gt_mask = skimage.io.imread(truth)
    pr_mask =  skimage.io.imread(pred)
    #pa = Panoptic_quality(gt_mask,pr_mask)
    dist_map = get_all_distance(gt_mask, label_dic, dist)
    weight_map = np.exp(-alpha*dist_map)
    iou = wIoU_evaluation(label_dic, gt_mask, pr_mask, weight_map)
    pa = Panoptic_quality(gt_mask,pr_mask,iou)
    if float(pa[0]) !=0:
        allpqs.append(float(pa[0]))
    if float(pa[1]) !=0:
        allsq.append(float(pa[1]))
    if float(pa[2]) !=0:
        allrq.append(float(pa[2]))
    print(i,len(alltruth),pa)
    i +=1
print("PQw:",np.mean(np.array(allpqs)))
print("SQw:",np.mean(np.array(allsq)))
print("RQw:",np.mean(np.array(allrq)))
