## BFF R-CNN: Balanced Feature Fusion for Object Detection ##
**usage：** 

1.install mmdetection

2.put mmdet and config folder under mmdetection，and replace the __init__.py.

3.run with config/BFF.py


**usage：** 

1.install mmdetection

2.copy mmdet and config to corresponding folder，cover init

3.in config use BFF.py instead

Overview of BFF, It consists of two parts: a, b and c are feature fusion from
the two ends to the center and then to each layer; d and f is Multilevel Region of Interest Features Extraction
(MRoIE). When the BFF method is used, on the MS COCO dataset, the AP is increased by 3.3 percentage points with respect to the performance of the baseline network.

![image](https://github.com/wangningwei12138/BFF-RCNN/blob/master/mmdet_BFF/architecture.png)   
Experiment on coco dataset


Table 1 Compared with the Faster RCNN on MS COCO datasets

| Method                                | *mAP* | *AP50* | *AP75* | *APs* | *APm* | *APl* |
|---------------------------------------|-------|--------|--------|-------|-------|-------|
| Baseline                              | 37.4  | 58.1   | 40.4   | 21.2  | 41.0  | 48.1  |
| PANet                                 | 37.5  | 58.6   | 40.8   | 21.5  | 41.0  | 48.6  |
| BEtM(ours)                            | 38.1  | 58.4   | 41.3   | 21.1  | 41.6  | 49.6  |
| BEtM + MRoIE(ours)                    | 39.3  | 60.2   | 42.8   | 23.3  | 42.9  | 50.6  |
| BFF +AWS (ours)                       | 39.5  | 60.3   | 42.6   | 23.1  | 43.1  | 50.6  |

Table 2 Compared experiments apply BFF to other networks

| Method                               | *mAP* | *AP50* | *AP75* | *APs* | *APm* | *APl* |
|--------------------------------------|-------|--------|--------|-------|-------|-------|
| Baseline                             | 37.4  | 58.1   | 40.4   | 21.2  | 41.0  | 48.1  |
| Libra-RCNN                           | 38.7  | 59.9   | 42.0   | 22.5  | 41.1  | 48.7  |
| Grid RCNN                            | 40.4  | 58.5   | 43.6   | 22.7  | 43.9  | 53.0  |
| Guided Anchoring                     | 39.6  | 58.7   | 43.4   | 21.2  | 43.0  | 52.7  |
| GRoIE                                | 37.5  | 59.2   | 40.6   | 22.3  | 41.5  | 47.8  |
| Libra-RCNN + BFF (ours)              | 39.5  | 59.4   | 43.0   | 22.8  | 42.8  | 51.3  |
| Grid RCNN + BFF (ours)               | 40.7  | 59.7   | 43.8   | 24.2  | 44.7  | 52.7  |
| Guided Anchoring + BFF (ours)        | 40.6  | 59.5   | 44.1   | 23.2  | 44.2  | 54.0  |
| GRoIE + BFF (ours)                   | 40.7  | 61.1   | 43.7   | 23.4  | 44.8  | 53.2  |
| Guided Anchoring+ GRoIE + BFF (ours) | 40.9  | 60.4   | 44.7   | 23.9  | 44.2  | 53.7  |


