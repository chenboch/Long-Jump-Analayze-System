# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import time
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log
from mmcv.transforms import Compose
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def process_one_image(args,
                      img,
                      detector,
                      test_pipeline,
                      pose_estimator,
                      tracker,
                      visualizer=None,
                      show_interval=0):
    result = inference_detector(detector, img,test_pipeline=test_pipeline)
    # Get candidate predict info with score threshold
    det_result = result.pred_instances[
        result.pred_instances.scores > args.score_thr].cpu().numpy()
    pred_instance = det_result
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[pred_instance.labels == args.det_cat_id]
    
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]
    # print(bboxes)
    new_bboxes = np.zeros((bboxes.shape[0],6))
    new_bboxes[:, :4] = bboxes
    new_bboxes[:, -2] = 0.9
    new_bboxes[:, -1] = 0
    online_targets = tracker.update(new_bboxes,img.copy())
    online_ids = []
    online_bbox = []
    for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
        if tlwh[2] * tlwh[3] > 10 :
            # tracker assigned id
            online_bbox.append(tlwh)
            online_ids.append(tid)
    new_online_box = []
    for bbox in online_bbox:
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
        new_online_box.append([x1, y1, x2, y2])
    bboxes = np.array(new_online_box)


    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None), online_ids
    