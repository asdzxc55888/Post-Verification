import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
from tqdm import tqdm

from networks import network
from config import cfg
from dataloader.VideosInfoLoader import VideosInfoLoader
from utils_local.osutils import mkdir_p, isdir


class keypointInfoHelper:
    '''
    Predicting keypoint of infant video
    '''
    def __init__(self, __video_name) -> None:
        self.__model = self.__initModel()
        self.__video_name = __video_name
        self.__video_loader = torch.utils.data.DataLoader(VideosInfoLoader(cfg, __video_name),
                                                        batch_size=36, shuffle=False, num_workers=8,
                                                        pin_memory=True)

    def __initModel(self) -> None:
        # image size torch.Size([128, 3, 384, 288]) ([batch, channels, width, height])
        __model = network.__dict__[cfg.model](
            cfg.output_shape, cfg.num_class, pretrained=False)
        __model = torch.nn.DataParallel(__model).cuda()

        checkpoint_file = os.path.join(cfg.checkpoint_path, 'CPN101_384x288.pth.tar')
        checkpoint = torch.load(checkpoint_file)
        __model.load_state_dict(checkpoint['state_dict'], False)
        # change to evaluation mode
        __model.eval()
        return __model

    def __saveResult(self, result_df: pd.DataFrame) -> None:
        '''
        Saving the video with keypoints information.
        '''
        if not isdir(cfg.keypoint_result_path):
            mkdir_p(cfg.keypoint_result_path)
        result_path = os.path.join(
            cfg.keypoint_result_path, self.__video_name.split('.')[0] + '.csv')
        result_df.to_csv(result_path, index=False)

    def predict(self) -> pd.DataFrame:
        '''
        Predicting the keypoints of the video and saving the mp4 type video.

        ### Result
        Keypoints predicted result: `DataFrame`
        '''
        full_result = []
        for i, (inputs, meta) in tqdm(enumerate(self.__video_loader)):
            with torch.no_grad():
                input_var = torch.autograd.Variable(inputs.cuda())
                # compute output
                global_outputs, refine_output = self.__model(input_var)
                score_map = refine_output.data.cpu()
                score_map = score_map.numpy()

                for b in range(inputs.size(0)):
                    details = meta['augmentation_details']
                    framesKey = meta['framesKey']
                    single_result_dict = {}
                    single_result = []

                    single_map = score_map[b]
                    for p in range(17):
                        single_map[p] /= np.amax(single_map[p])
                        border = 10
                        dr = np.zeros(
                            (cfg.output_shape[0] + 2*border, cfg.output_shape[1]+2*border))
                        dr[border:-border, border:-border] = single_map[p].copy()
                        dr = cv2.GaussianBlur(dr, (21, 21), 0)
                        lb = dr.argmax()
                        y, x = np.unravel_index(lb, dr.shape)
                        dr[y, x] = 0
                        lb = dr.argmax()
                        py, px = np.unravel_index(lb, dr.shape)
                        y -= border
                        x -= border
                        py -= border + y
                        px -= border + x
                        ln = (px ** 2 + py ** 2) ** 0.5
                        delta = 0.25
                        if ln > 1e-3:
                            x += delta * px / ln
                            y += delta * py / ln
                        x = max(0, min(x, cfg.output_shape[1] - 1))
                        y = max(0, min(y, cfg.output_shape[0] - 1))
                        resy = float(
                            (4 * y + 2) / cfg.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
                        resx = float(
                            (4 * x + 2) / cfg.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
                        single_result.append([resx, resy])
                    if len(single_result) != 0:
                        single_result_dict['framesKey'] = framesKey[b].item()
                        single_result_dict['keypoints'] = single_result
                        full_result.append(single_result_dict)

        result_df = pd.DataFrame(full_result, columns=[
            'framesKey', 'keypoints'])
        self.__saveResult(result_df)
        return result_df
