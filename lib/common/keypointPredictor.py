import os
import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
from tqdm import tqdm
import torch.backends.cudnn as cudnn

from config import cfg
from dataloader.VideosInfoLoader import VideosInfoLoader
from utils.osutils import mkdir_p, isdir
from core.function import get_final_preds

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class keypointPredictor:
    '''
    Predicting keypoint of infant video
    '''

    def __init__(self, video_name) -> None:
        self.__model = self.__initModel()
        self.__video_name = video_name
        self.__video_loader = torch.utils.data.DataLoader(VideosInfoLoader(cfg, video_name),
                                                          batch_size=cfg.TEST.BATCH_SIZE_PER_GPU, shuffle=False, num_workers=cfg.WORKERS,
                                                          pin_memory=True)

    def __initModel(self) -> None:
        cudnn.benchmark = cfg.CUDNN.BENCHMARK 
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
        
        model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
            cfg, is_train=False
        )
        model.load_state_dict(torch.load(
            cfg.TEST.MODEL_FILE), strict=False)
        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

        return model

    def __saveResult(self, result_df: pd.DataFrame) -> None:
        '''
        Saving the video with keypoints information.
        '''
        if not isdir(cfg.KEYPOINT_RESULT_PATH):
            mkdir_p(cfg.KEYPOINT_RESULT_PATH)
        result_path = os.path.join(
            cfg.KEYPOINT_RESULT_PATH, self.__video_name.split('.')[0] + '.csv')
        result_df.to_csv(result_path, index=False)


    def predict(self) -> pd.DataFrame:
        '''
        Predicting the keypoints of the video and saving the mp4 type video.

        ### Result
        Keypoints predicted result: `DataFrame`
        '''
        full_result = []
        self.__model.eval()
        for i, (inputs, meta) in tqdm(enumerate(self.__video_loader)):
            framesKey = meta['framesKey']
            centers = meta['center'].numpy()
            scales = meta['scale'].numpy()

            input_var = torch.autograd.Variable(inputs.cuda())

            with torch.no_grad():
                outputs = self.__model(input_var)
                preds, _ = get_final_preds(
                    cfg,
                    outputs.clone().cpu().numpy(),
                    centers,
                    scales)
            
            for b in range(inputs.size(0)):
                single_result_dict = {}
                single_result_dict['framesKey'] = framesKey[b].item()
                single_result_dict['keypoints'] = preds[b].tolist()
                full_result.append(single_result_dict)

        result_df = pd.DataFrame(full_result, columns=[
            'framesKey', 'keypoints'])
        self.__saveResult(result_df)
        return result_df
