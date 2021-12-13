import pandas as pd
import torch
from math import floor
from tqdm import tqdm
import cv2

class ImageBboxInfo:
    '''
    Use pretrained yolov5 model to predict bbox position info of the human in the video.
    '''
    def __init__(self, video_path, batch_size = 16):
        self.video_path = video_path
        self.batch_size = batch_size
        self.frames = self.__getVideoFrames()
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.result = self.__predictVideoBboxes()

    def __getVideoFrames(self):
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def __predictVideoBboxes(self):
        print("predict Video bboxes...")
        result = None
        batch_size = self.batch_size
        df= pd.DataFrame({})

        for epoch in tqdm(range(floor(len(self.frames)/batch_size + 1))):
            start_index = epoch * batch_size
            end_index = epoch * batch_size + batch_size
            result = self.model(self.frames[start_index:end_index])

            for i in range(len(result.pandas().xyxy)):
                singalDf = result.pandas().xyxy[i]
                singalDf['framesKey'] = i + start_index
                df = df.append(singalDf)

        isOverBaseConfidence = df["confidence"] > 0.5
        isPersonClass = df["class"] == 0

        filter_df = df[(isOverBaseConfidence & isPersonClass )]
        return filter_df

    def getVideoInfo(self):
        '''
        Get video info with dataframe(pandas) type
        '''
        return self.result



        
