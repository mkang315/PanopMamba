import numpy as np
import torch

def panoptic_quality(ground_truth_image, predicted_image):
    # 确保将张量从 GPU 移动到 CPU
    ground_truth_image = ground_truth_image.cpu().numpy()  # 确保在 CPU 上
    predicted_image = predicted_image.cpu().numpy()  # 确保在 CPU 上

    TP = 0
    FP = 0
    FN = 0
    sum_IOU = 0
    matched_instances = {}

    # Find matched instances and save it in a dictionary
    for i in np.unique(ground_truth_image):
    # for i in np.unique(ground_truth_image.cpu()):

        if i == 0:
            continue
        temp_image = ground_truth_image == i
        matched_image = temp_image * predicted_image

        for j in np.unique(matched_image):
            if j == 0:
                continue
            pred_temp = predicted_image == j
            intersection = np.sum(temp_image * pred_temp)
            union = np.sum(temp_image + pred_temp)
            IOU = intersection / (union + 1e-6)  # 避免除零
            if IOU > 0.5:
                matched_instances[i] = (j, IOU)

    # Compute TP, FP, FN and sum of IOU of the matched instances to compute Panoptic Quality
    pred_indx_list = np.unique(predicted_image)[1:]  # Remove 0 from the predicted instances

    for indx in np.unique(ground_truth_image):
        if indx == 0:
            continue
        if indx in matched_instances.keys():
            pred_indx_list = np.delete(pred_indx_list, np.argwhere(pred_indx_list == matched_instances[indx][0]))
            TP += 1
            sum_IOU += matched_instances[indx][1]
        else:
            FN += 1

    FP = len(pred_indx_list)
    PQ = sum_IOU / (TP + 0.5 * FP + 0.5 * FN) if (TP + 0.5 * FP + 0.5 * FN) > 0 else 0

    return PQ
