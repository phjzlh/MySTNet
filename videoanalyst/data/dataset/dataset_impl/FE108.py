# -*- coding: utf-8 -*-
import copy
import os.path as osp
from typing import Dict, List

import numpy as np
from loguru import logger

from videoanalyst.data.dataset.dataset_base import TRACK_DATASETS, DatasetBase
from videoanalyst.evaluation.got_benchmark.datasets import GOT10k
from videoanalyst.pipeline.utils.bbox import xywh2xyxy

_current_dir = osp.dirname(osp.realpath(__file__))


@TRACK_DATASETS.register
class FE108Dataset(DatasetBase):
    r"""
    FE108 dataset helper

    Hyper-parameters
    ----------------
    dataset_root: str
        path to root of the dataset
    subset: str
        dataset split name (train|val|test)
    ratio: float
        dataset ratio. used by sampler (data.sampler).
    max_diff: int
        maximum difference in index of a pair of sampled frames
    check_integrity: bool
        if check integrity of dataset or not
    """
    default_hyper_params = dict(
        dataset_root="datasets/FE108",
        subset="train",
        ratio=1.0,
        max_diff=100,
        check_integrity=True,
    )

    def __init__(self) -> None:
        r"""
        Create dataset with config

        Arguments
        ---------
        cfg: CfgNode
            dataset config
        """
        super(FE108Dataset, self).__init__()
        self._state["dataset"] = None

    def update_params(self):
        r"""
        an interface for update params
        """
        dataset_root = osp.realpath(self._hyper_params["dataset_root"])
        subset = self._hyper_params["subset"]
        check_integrity = self._hyper_params["check_integrity"]
        self._state["dataset"] = GOT10k(dataset_root,
                                        subset=subset,
                                        check_integrity=check_integrity)

    def __getitem__(self, item: int) -> Dict:
        img_files_1, img_files_2, anno = self._state["dataset"][item]

        anno = xywh2xyxy(anno)
        sequence_data_pos = dict(image_1=img_files_1, anno=anno)
        sequence_data_neg = dict(image_2 = img_files_2, anno=anno)

        return sequence_data_pos, sequence_data_neg

    def __len__(self):
        return len(self._state["dataset"])

