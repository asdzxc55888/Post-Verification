# Post-Verification

## 實驗目的
以孩童在影片中移動的變動量，辨識孩童粗大動作的發展年齡
- [x] 偵測孩童在影像中的位置
- [x] 辨識孩童的關節點位置
- [x] 計算孩童關節點變動量
- [ ] 訓練or分類孩童粗大動作年齡層
- [ ] 串接至ONE發展評估系統
- [ ] 能將分類完的結果顯示至系統

## 前處理流程

1. 透過fasterrcnn_resnet50_fpn預測人體關節點
2. 取得人體位置後透過HR-Net預測人體關節點
3. 計算人體關節點變動量
4. 將正規化變動量

## 檔案結構說明
```
.
├── checkpoint
│   └── pose_hrnet_w48_384x288.pth // HR-Net pre-train model
│
├── data
│   ├── medias // 放置孩童影片
│   └── video_info // 2d keypoint分析結果
├── dataloader
│   └── VideoInfoLoader.py //讀取影片預測box info資訊
│   
├── lib
│   ├── common //存
│       ├── bboxInfoPredictor.py //輸入影片輸出影片box位置
│       ├── keypointPredictor.py //輸入影片與影片box位置輸出關節點位置
│       ├── humanKeypoints.py //讀取已預測後的關節點資料
│       └── visualization.py //將關節點視覺化
│   ├── config  //處理模型與存放位置設定檔案
│   ├── core    //
│   ├── models  //HR-NET model
│   └── utils   //
├── output
│   ├── boxes_info        //預測box info檔案存放資料夾
│   ├── keypoint_results  //關節點預測結果
│   └── video_results     //附帶關節點的影片輸出結果
└── demo.ipn
```

## 參考
[FASTERRCNN_RESNET50_FPN](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html)
[Deep High-Resolution Representation Learning
for Visual Recognition](https://arxiv.org/pdf/1908.07919.pdf)
[HRNet-Human-Pose-Estimation](https://github.com/HRNet/HRNet-Human-Pose-Estimation)