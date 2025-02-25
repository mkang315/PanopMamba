# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image
from prettytable import PrettyTable
from .panoptic_quality import panoptic_quality
from mmseg.registry import METRICS


@METRICS.register_module()
class IoUMetric(BaseMetric):
    """IoU evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 ignore_index: int = 255,
                 iou_metrics: List[str] = ['mIoU'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        num_classes = len(self.dataset_meta['classes'])
        for data_sample in data_samples:
            name = data_sample['seg_map_path'].split(os.sep)[-1].split(".")[0]
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            cv2.imwrite(f"./txtoutputs/predict/{name}.png",pred_label.cpu().detach().numpy())
            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample['gt_sem_seg']['data'].squeeze().to(
                    pred_label)
                cv2.imwrite(f"./txtoutputs/real/{name}.png",label.cpu().detach().numpy())
                self.results.append(
                    self.intersect_and_union(pred_label, label, num_classes,
                                             self.ignore_index))
            # print(self.results)
            # format_result
            if self.output_dir is not None:
                basename = osp.splitext(osp.basename(
                    data_sample['img_path']))[0]
                png_filename = osp.abspath(
                    osp.join(self.output_dir, f'{basename}.png'))
                output_mask = pred_label.cpu().numpy()
                # The index range of official ADE20k dataset is from 0 to 150.
                # But the index range of output is from 0 to 149.
                # That is because we set reduce_zero_label=True.
                if data_sample.get('reduce_zero_label', False):
                    output_mask = output_mask + 1
                output = Image.fromarray(output_mask.astype(np.uint8))
                output.save(png_filename)


    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        results = tuple(zip(*results))




        assert len(results) == 5

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        # print(len(results[3]))
        # pq = results[4]
        print(len(results[4]))
        print("mpq: {:.2f}%".format(np.mean(results[4]) * 100))

        ret_metrics = self.total_area_to_metrics(
            total_area_intersect, total_area_union, total_area_pred_label,
            total_area_label, self.metrics, self.nan_to_num, self.beta)

        class_names = self.dataset_meta['classes']

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                metrics[key] = val
            else:
                metrics['m' + key] = val

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()

        for key, val in ret_metrics_class.items():
            # if np.isscalar(val):
            #     val = [val]  # Convert scalar to a list
            class_table_data.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics

    @staticmethod
    def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                            num_classes: int, ignore_index: int):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """
        pq = panoptic_quality(label, pred_label)

        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]


        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_pred_label = torch.histc(
            pred_label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_label = torch.histc(
            label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label,pq

    @staticmethod
    def total_area_to_metrics(total_area_intersect: np.ndarray,
                              total_area_union: np.ndarray,
                              total_area_pred_label: np.ndarray,
                              total_area_label: np.ndarray,
                              metrics: List[str] = ['mIoU'],
                              nan_to_num: Optional[int] = None,
                              beta: int = 1):
        """Calculate evaluation metrics
        Args:
            total_area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            total_area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            total_area_label (np.ndarray): The ground truth histogram on
                all classes.
            metrics (List[str] | str): Metrics to be evaluated, 'mIoU', 'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be
                replaced by the numbers defined by the user. Default: None.
            beta (int): Determines the weight of recall in the combined score.
                Default: 1.
        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """

        def f_score(precision, recall, beta=1):
            """calculate the f-score value.

            Args:
                precision (float | torch.Tensor): The precision value.
                recall (float | torch.Tensor): The recall value.
                beta (int): Determines the weight of recall in the combined
                    score. Default: 1.

            Returns:
                [torch.tensor]: The f-score value.
            """
            score = (1 + beta ** 2) * (precision * recall) / (
                    (beta ** 2 * precision) + recall)
            return score

        # def calculate_pq(intersect, union, pred_label, gt_label):
        #     """Calculate Panoptic Quality (PQ).
        #     PQ = IoU * RQ (Recognition Quality) * SQ (Segmentation Quality)
        #
        #     Args:
        #         intersect (np.ndarray): Intersection values for each class.
        #         union (np.ndarray): Union values for each class.
        #         pred_label (np.ndarray): Prediction histogram.
        #         gt_label (np.ndarray): Ground truth histogram.
        #
        #     Returns:
        #         np.ndarray: PQ score for each class.
        #     """
        #     # Segmentation Quality (SQ)
        #     sq = intersect / union
        #     # Recognition Quality (RQ)
        #     rq = intersect / (pred_label + gt_label - intersect)
        #     pq = sq * rq
        #     return pq
        # def calculate_pq(total_area_intersect, total_area_union, total_area_pred_label, total_area_label):
        #     """Calculate Panoptic Quality (PQ).
        #
        #     Args:
        #         total_area_intersect (list): Intersection values for each class.
        #         total_area_union (list): Union values for each class.
        #         total_area_pred_label (list): Predictions histogram for each class.
        #         total_area_label (list): Ground truth histogram for each class.
        #
        #     Returns:
        #         list: PQ scores for each class.
        #     """
        #     pq_scores = []
        #
        #     for tp, union, pred, label in zip(total_area_intersect, total_area_union, total_area_pred_label,
        #                                       total_area_label):
        #         tp = tp.cpu().numpy() if isinstance(tp, torch.Tensor) else np.array(tp)
        #         union = union.cpu().numpy() if isinstance(union, torch.Tensor) else np.array(union)
        #         pred = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else np.array(pred)
        #         label = label.cpu().numpy() if isinstance(label, torch.Tensor) else np.array(label)
        #
        #         fp = pred - tp
        #         fn = label - tp
        #
        #         sum_IOU = np.sum(tp) / (np.sum(union) + 1e-6)  # 避免除零
        #
        #         denominator = np.sum(tp) + 0.5 * np.sum(fp) + 0.5 * np.sum(fn)
        #         pq = sum_IOU / denominator if denominator > 0 else 0
        #
        #         pq_scores.append(pq)
        #         pq_scores_tensor = torch.tensor(pq_scores)
        #
        #     return pq_scores_tensor
        def calculate_pq(total_area_intersect, total_area_union, total_area_pred_label, total_area_label,
                         iou_threshold=0.5):
            """Calculate Panoptic Quality (PQ).

            Args:
                total_area_intersect (list): Intersection values for each class.
                total_area_union (list): Union values for each class.
                total_area_pred_label (list): Predictions histogram for each class.
                total_area_label (list): Ground truth histogram for each class.
                iou_threshold (float): IoU threshold for considering a match.

            Returns:
                list: PQ scores for each class.
            """
            pq_scores = []

            for tp, union, pred, label in zip(total_area_intersect, total_area_union, total_area_pred_label,
                                              total_area_label):
                tp = tp.cpu().numpy() if isinstance(tp, torch.Tensor) else np.array(tp)
                union = union.cpu().numpy() if isinstance(union, torch.Tensor) else np.array(union)
                pred = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else np.array(pred)
                label = label.cpu().numpy() if isinstance(label, torch.Tensor) else np.array(label)

                fp = pred - tp
                fn = label - tp

                # 计算 IoU
                sum_IOU = np.sum(tp) / (np.sum(union) + 1e-6)  # 避免除零

                # 仅在 IoU 大于阈值时才计算 PQ
                if sum_IOU > iou_threshold:
                    denominator = np.sum(tp) + 0.5 * np.sum(fp) + 0.5 * np.sum(fn)
                    pq = sum_IOU / denominator if denominator > 0 else 0
                else:
                    pq = 0

                pq_scores.append(pq)

            pq_scores_tensor = torch.tensor(pq_scores)
            return pq_scores_tensor

        # def calculate_pq(total_area_intersect, total_area_union, total_area_pred_label, total_area_label):
        #     """Calculate Panoptic Quality (PQ).
        #
        #     Args:
        #         total_area_intersect (torch.Tensor): Intersection of predictions and ground truth.
        #         total_area_union (torch.Tensor): Union of predictions and ground truth.
        #         total_area_pred_label (torch.Tensor): Predictions.
        #         total_area_label (torch.Tensor): Ground truth.
        #
        #     Returns:
        #         float: Panoptic Quality (PQ) score.
        #     """
        #     tp = total_area_intersect  # True Positives
        #     fp = total_area_pred_label - total_area_intersect  # False Positives
        #     fn = total_area_label - total_area_intersect  # False Negatives
        #
        #     # To avoid division by zero
        #     iou = total_area_intersect / (total_area_union + 1e-6)  # 加小值避免除零
        #     pq = np.sum(iou.cpu().numpy()) / (tp.sum().item() + 0.5 * fp.sum().item() + 0.5 * fn.sum().item()) if (
        #                                                                                                                       tp.sum() + 0.5 * fp.sum() + 0.5 * fn.sum()) > 0 else 0
        #
        #     return pq

        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore', 'PQ']
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f'metrics {metrics} is not supported')

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({'aAcc': all_acc})
        for metric in metrics:
            if metric == 'mIoU':
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics['IoU'] = iou
                ret_metrics['Acc'] = acc
            elif metric == 'mDice':
                dice = 2 * total_area_intersect / (
                        total_area_pred_label + total_area_label)
                acc = total_area_intersect / total_area_label
                ret_metrics['Dice'] = dice
                ret_metrics['Acc'] = acc
            elif metric == 'mFscore':
                precision = total_area_intersect / total_area_pred_label
                # print(precision)
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor([
                    f_score(x[0], x[1], beta) for x in zip(precision, recall)
                ])
                ret_metrics['Fscore'] = f_value
                ret_metrics['Precision'] = precision
                ret_metrics['Recall'] = recall
            elif metric == 'PQ':
                pq = calculate_pq(total_area_intersect, total_area_union, total_area_pred_label, total_area_label)
                # print(pq)
                ret_metrics['PQ'] = pq

        ret_metrics = {
            metric: value.numpy()
            # metric: value.cpu().numpy() if isinstance(value, torch.Tensor) else value
            for metric, value in ret_metrics.items()
        }


        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })
        return ret_metrics
