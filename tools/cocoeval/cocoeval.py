# -*- coding: UTF-8 -*-

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if __name__ == '__main__':
    pred_json = './pred_coco.json'
    anno_json = './gt_coco.json'

    # 使用COCO API加载预测结果和标注
    cocoGt = COCO(anno_json)
    cocoDt = cocoGt.loadRes(pred_json)

    # 创建COCOeval对象
    cocoEval = COCOeval(cocoGt, cocoDt, 'segm') # bbox  segm

    # 执行评估
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()



