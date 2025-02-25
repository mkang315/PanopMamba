import glob

import numpy as np
import skimage
from PIL import Image
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        return IoU

    def dice(self):
        dice = 2 * np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0)
        )

        return dice

    def Mdice(self):
        dice = 2 * np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0)
        )
        Mdice = np.nanmean(dice)
        return Mdice

    def Frequency_Weighted_Intersection_over_Union(self,isTotal=False):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        # if isTotal:
        #     freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        # else:
        #     freq = np.sum(self.confusion_matrix, axis=1)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        if isTotal:
            FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
            return FWIoU
        else:
            return  iu
    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)

        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)




def Panoptic_quality(ground_truth_image,predicted_image,ious):
    TP = 0
    FP = 0
    FN = 0
    sum_IOU = 0
    matched_instances = {}
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
                    #pred_temp = predicted_image == j
                    # intersection = sum(sum(temp_image*pred_temp))
                    # union = sum(sum(temp_image + pred_temp))
                    IOU =  ious[i]
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

totals = Evaluator(5)
index= -1

allpqs = []
allsq = []
allrq = []
allious =[]

for truth,pred in zip(alltruth,allpred):
    index+=1
    gt_img = skimage.io.imread(truth)
    pre_img =  skimage.io.imread(pred)
    pre_img=np.asarray(pre_img)
    gt_img =np.asarray(gt_img)
    test = Evaluator(5)
    test.confusion_matrix=test._generate_matrix(gt_img,pre_img)
    fwIoUs =test.Frequency_Weighted_Intersection_over_Union(isTotal=False)
    #print(fwIoUs)
    pa = Panoptic_quality(gt_img,pre_img,fwIoUs)
    if float(pa[0]) !=0:
        allpqs.append(float(pa[0]))
    if float(pa[1]) !=0:
        allsq.append(float(pa[1]))
    if float(pa[2]) !=0:
        allrq.append(float(pa[2]))

    if index==0:
        totals.confusion_matrix=totals._generate_matrix(gt_img,pre_img)
    else:
        totals.add_batch(gt_img,pre_img)


FWIoU =totals.Frequency_Weighted_Intersection_over_Union(isTotal=True)

print("FWIoU:",FWIoU)
print("PQfw:",np.mean(np.array(allpqs)))
print("SQfw:",np.mean(np.array(allsq)))
print("RQfw:",np.mean(np.array(allrq)))

# print("Overall FWIoU:",FWIoU)
# print("ACC: ",val3)
# print("MIou: ",val2)
# print("Iou",val4)
# print("dice: ",val5)
# print("mdice",val6)


