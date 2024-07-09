"""
@author: yury
@date: 20240628
@contact: 495978477@qq.com
"""

import sys
import os

sys.path.append('/data/ai/faceSDK')
import logging

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
import logging.config

logging.config.fileConfig("/data/ai/faceSDK/config/logging.conf")
logger = logging.getLogger('api')

import time
import cv2
import numpy as np
import pandas as pd
import faiss
from sklearn.preprocessing import normalize
import torch
#torch.nn.Module.dump_patches = True

from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
from models.model_pipline import ModelLoader

"""
    需求：读取数据库照片，提取每张照片的人脸特征向量保存在faiss向量数据库中，并且数据库的索引与照片的名字一一映射
"""

if __name__ == '__main__':
    t0 = time.time()

    # 初始化模型加载器，加载模型
    model_path = '/data/ai/faceSDK/models'
    model_loader = ModelLoader(model_path)

    # 获取需要的模型处理器
    faceDetModelHandler = model_loader.get_face_det_model_handler()
    faceAlignModelHandler = model_loader.get_face_align_model_handler()
    faceRecModelHandler = model_loader.get_face_rec_model_handler()
    face_cropper = FaceRecImageCropper()

    # 读取存放图片路径
    image_path = '/data/ai/faceSDK/api_usage/facebase_img'
    image_list = os.listdir(image_path)

    # 读取 id 和对应的姓名
    name_file = '/data/ai/faceSDK/api_usage/img.xlsx'
    df = pd.read_excel(name_file, dtype={'id': str})
    id_to_name = {row['id']: row['name'] for _, row in df.iterrows()}  # '001': 'xxxx'
    #print(id_to_name)

    feature_list = []
    index_to_name = {}

    for i, each in enumerate(image_list):
        image = cv2.imread(os.path.join(image_path, each), cv2.IMREAD_COLOR)

        try:
            # -----  人脸特征提取  ------
            dets = faceDetModelHandler.inference_on_image(image)  # 人脸检测
            if dets.shape[0] != 1:
                logger.info('Input image should contain only one faces!')

            landmarks = faceAlignModelHandler.inference_on_image(image, dets[0])  # 人脸关键点识别

            landmarks_list = []
            for (x, y) in landmarks.astype(np.int32):
                landmarks_list.extend((x, y))

            cropped_image = face_cropper.crop_image_by_mat(image, landmarks_list)  # 人脸剪裁
            feature = faceRecModelHandler.inference_on_image(cropped_image)  # 人脸特征值提取
            feature_list.append(feature)

            # -----  将索引与姓名映射关系保存到字典中  -----
            idx = len(feature_list) - 1
            image_id = each.split('.')[0]  # 获取图片的 ID，例如 '001'
            index_to_name[idx] = id_to_name.get(image_id, 'Unknown')  # "1":"xxxx"

            logger.info('Successfully extracted the %d face feature from %s' % (i, each))

        except Exception as e:
            logger.error('Failed to extract the %d face feature from %s' % (i, each))
            logger.error(e)
            sys.exit(-1)

    # 保存到本地
    #np.save('/data/ai/faceSDK/api_usage/facebase/feature_list.npy', feature_list)

    # 保存在faiss向量数据库中
    feature_matrix = np.array(feature_list)
    feature_matrix = normalize(feature_matrix, axis=1, norm='l2')

    # 特征向量的维度
    d = feature_matrix.shape[1]

    # 初始化Faiss索引
    index = faiss.IndexFlatL2(d)  # 使用欧氏距离（L2距离）作为度量方法
    index.add(feature_matrix.astype(np.float32))

    # 保存Faiss索引到磁盘
    index_file = "/data/ai/faceSDK/api_usage/facebase/faiss_index"
    faiss.write_index(index, index_file)

    # 保存索引到姓名字典的映射
    index_to_name_file = "/data/ai/faceSDK/api_usage/facebase/index_to_name_mapping.npy"
    print(index_to_name)
    np.save(index_to_name_file, index_to_name)

    print('Done. (%.3fs)' % (time.time() - t0))
    print("FaceDataBase Initial Success!")
