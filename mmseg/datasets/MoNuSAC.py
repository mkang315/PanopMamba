import os.path as osp

import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class MoNuSACDataset(BaseSegDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    METAINFO = dict(
        #  'Epithelial', 'Lymphocyte', 'Neutrophil', 'Macrophage'
        classes=('background', 'Epithelial', 'Lymphocyte', 'Neutrophil','Macrophage'),
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
             [0, 0, 128]])
       # palette=[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])

    def __init__(self,
                 ann_file,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            **kwargs)
        assert fileio.exists(self.data_prefix['img_path'],
                             self.backend_args) and osp.isfile(self.ann_file)
