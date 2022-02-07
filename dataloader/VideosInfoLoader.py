import os
import numpy as np
import json
import random
import cv2

import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import pandas as pd

import sys
sys.path.append(os.getcwd()+"/..")

from lib.utils.osutils import *
from lib.utils.imutils import *
from lib.utils.transforms import *
from lib.config import cfg
from lib.utils.transforms import get_affine_transform

# import imageio

class VideosInfoLoader(data.Dataset):
    def __init__(self, cfg, video_name):
        self.video_path = os.path.join(cfg.VIDEO_PATH, video_name)
        self.bbox_infos = pd.read_csv(os.path.join(cfg.BBOX_INFO_PATH, video_name.split(".")[0] + ".csv"))
        self.frames = self.__getVideoFrames()

    def __box_to_center_scale(self, box, model_image_width, model_image_height):
        """convert a box to center,scale information required for pose transformation
        Parameters
        ----------
        box : list of tuple
            list of length 2 with two tuples of floats representing
            bottom left and top right corner of a box
        model_image_width : int
        model_image_height : int
    
        Returns
        -------
        (numpy array, numpy array)
            Two numpy arrays, coordinates for the center of the box and the scale of the box
        """
        center = np.zeros((2), dtype=np.float32)
    
        bottom_left_corner = box[0]
        top_right_corner = box[1]
        box_width = top_right_corner[0]-bottom_left_corner[0]
        box_height = top_right_corner[1]-bottom_left_corner[1]
        bottom_left_x = bottom_left_corner[0]
        bottom_left_y = bottom_left_corner[1]
        center[0] = bottom_left_x + box_width * 0.5
        center[1] = bottom_left_y + box_height * 0.5
    
        aspect_ratio = model_image_width * 1.0 / model_image_height
        pixel_std = 200
    
        if box_width > aspect_ratio * box_height:
            box_height = box_width * 1.0 / aspect_ratio
        elif box_width < aspect_ratio * box_height:
            box_width = box_height * aspect_ratio
        scale = np.array(
            [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25
    
        return center, scale
  
    def __getitem__(self, key):
        video_info = self.bbox_infos.iloc[key]
        frameKey = video_info['framesKey']
        gt_bbox = eval(video_info['boxes'])
        image = self.frames[frameKey]

        rotation = 0
        center, scale = self.__box_to_center_scale(gt_bbox, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
        # pose estimation transformation
        trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
        model_input = cv2.warpAffine(
            image,
            trans,
            (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # pose estimation inference
        model_input = transform(model_input)
        meta = {'framesKey' : frameKey, 'center': center, 'scale':scale}

        return model_input, meta

    def __getVideoFrames(self):
        frames = []
        cap = cv2.VideoCapture(self.video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


        return frames

    def __len__(self):
        return self.bbox_infos.shape[0]

