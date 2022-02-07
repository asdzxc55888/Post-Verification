import string
import pandas as pd
import torch
import numpy as np
from math import floor
from tqdm import tqdm
import cv2
import os

import torchvision
from lib.utils.osutils import mkdir_p, isdir
from lib.config import cfg

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

CTX = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


class bboxInfoHelper:
    '''
    Use pretrained yolov5 model to predict bbox position info of the human in the video.
    '''

    def __init__(self, video_name: string, batch_size: int = 8, threshold=0.5):
        self.__video_name = video_name
        self.video_path = os.path.join(cfg.VIDEO_PATH, video_name)
        self.batch_size = batch_size
        # self.frames = self.__getVideoFrames()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True)
        self.model.to(CTX)
        self.threshold = threshold

    def __getVideoFrames(self) -> list:
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(img)
        cap.release()
        print("getVideoFrames finished")
        return frames

    def __saveInfoResult(self, result_df) -> None:
        if not isdir(cfg.KEYPOINT_RESULT_PATH):
            mkdir_p(cfg.KEYPOINT_RESULT_PATH)
        result_path = os.path.join(
            cfg.BBOX_INFO_PATH, self.__video_name.split(".")[0] + '.csv')
        result_df.to_csv(result_path, index=False)

    def predict(self) -> pd.DataFrame:
        print("predict Video bboxes...")
        result = None
        frames = self.__getVideoFrames()

        full_result = []
        self.model.eval()
        # for epoch in tqdm(range(floor(len(self.frames)/batch_size + 1))):
        for index in tqdm(range(len(frames))):
            # start_index = epoch * batch_size
            # end_index = epoch * batch_size + batch_size
            frame_np = np.array(frames[index])
            img_tensor = torch.from_numpy(
                frame_np/255.).permute(2, 0, 1).float().to(CTX)
            result = self.model([img_tensor])


            single_result = {}
            pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[j]
                            for j in list(result[0]['labels'].cpu().numpy())]  # Get thePrediction Score
            pred_boxes = [[(j[0], j[1]), (j[2], j[3])]
                          for j in list(result[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes
            pred_score = list(result[0]['scores'].detach().cpu().numpy())

            if len(pred_score) >= 1 and max(pred_score) >= self.threshold:
                pred_t = [pred_score.index(x)
                          for x in pred_score if x > self.threshold][-1]
                pred_boxes = pred_boxes[:pred_t+1]
                pred_classes = pred_classes[:pred_t+1]
                for idx, box in enumerate(pred_boxes):
                    if pred_classes[idx] == 'person':
                        single_result["framesKey"] = index
                        single_result["boxes"] = box
                        full_result.append(single_result)
                        break

            # for i in range(len(result.pandas().xyxy)):
            #     singalDf = result.pandas().xyxy[i]
            #     singalDf['framesKey'] = i + start_index
            #     df = df.append(singalDf)

        result_df = pd.DataFrame(full_result, columns=[
            'framesKey', 'boxes'])
        self.__saveInfoResult(result_df)

        return result_df

    def getVideoInfo(self) -> pd.DataFrame:
        '''
        Get video info with dataframe(pandas) type
        '''
        return self.result
