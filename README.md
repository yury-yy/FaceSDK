# 一、人脸对比
基于京东开源的FaceX-Zoo工具包实现人脸对比

# 环境
* python >= 3.7.1  
* pytorch >= 1.1.0  
* torchvision >= 0.3.0 
* yaml  
* itertools  
* skimage

# 使用方式
```python
python api_usage/feature_pipline.py   # 建立人脸库
python api_usage/recognise_pipline.py   # 人脸对比
```

# 二、人脸表情识别
基于京东开源的FaceX-Zoo工具包实现人脸检测、关键点描绘、人脸表情识别

# 使用方式
```python
python emotion/emotion_pipline.py -src emotion/test\
                --result_folder emotion/result  # 人脸表情识别
```
