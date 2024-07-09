"""
@author: yury
@date: 20240628
@contact: 495978477@qq.com
"""
import sys
sys.path.append('/data/ai/faceSDK')
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
import logging.config
logging.config.fileConfig("/data/ai/faceSDK/config/logging.conf")
logger = logging.getLogger('api')

import math
import time
import cv2
import numpy as np
import faiss
import torch
#torch.nn.Module.dump_patches = True

from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
from models.model_pipline import ModelLoader

"""
    需求：传入一张人脸照片。其中，该照片只包含一张人脸，尽量是正面照（识别准确度更高），与人脸库的数据比对，如果与人脸库的结果匹配成功，
    则输出相似度和对于名字，并且在图片上添加标签（名字+相似度）保存下来。
"""



"""
欧氏距离转百分比
一般80%左右就能确定是同一个人
"""
def eD2percentage(eD):
    a, b = 3.30, -3.93
    per = 1 / (1 + math.e ** (a * eD + b))
    per = int(per * 100.0)
    per_str = str(per) + "%"
    return per_str


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

    image_path = '/data/ai/faceSDK/api_usage/test_img/test01.jpg'  # 识别的单人照片
    save_path_img = '/data/ai/faceSDK/api_usage/result/test01.jpg'  # 识别保存结果路径
    index_to_name_file = "/data/ai/faceSDK/api_usage/facebase/index_to_name_mapping.npy"  # 索引到姓名映射字典
    faiss_index_file = "/data/ai/faceSDK/api_usage/facebase/faiss_index"   # 人脸特征向量数据库

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    index_to_name = np.load(index_to_name_file, allow_pickle=True).item()
    index = faiss.read_index(faiss_index_file)

    # 提取要识别的照片的人脸特征向量
    try:
        dets = faceDetModelHandler.inference_on_image(image)
        if dets.shape[0] != 1:
            logger.info('Input image should contain only one faces!')
        landmarks = faceAlignModelHandler.inference_on_image(image, dets[0])
        landmarks_list = []
        for (x, y) in landmarks.astype(np.int32):
            landmarks_list.extend((x, y))
        cropped_image = face_cropper.crop_image_by_mat(image, landmarks_list)
        feature = faceRecModelHandler.inference_on_image(cropped_image)

        logger.info('Success extracted face feature!')

    except Exception as e:
        logger.error('Failed extracted face feature.')
        logger.error(e)
        sys.exit(-1)

    # 计算最相似的人脸特征向量
    query_vector = np.array([feature.astype(np.float32)])
    D, I = index.search(query_vector, k=1)  # 查找最相似的一个向量

    # 获取最相似向量的索引及其对应的名字
    min_distance = D[0][0]
    distance_percent = eD2percentage(min_distance)  # 获取相似百分比
    index_faiss = I[0][0]

    # 根据最小欧氏距离判断匹配结果
    if min_distance < 1.45:
        matched_name = index_to_name[index_faiss]  # 获取匹配结果的标签
    else:
        matched_name = 'Unknown'

    # 在图像上绘制方框和标签
    box = dets[0]
    try:
        box = list(map(int, box))
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        label = matched_name + distance_percent
        #tl = 3 or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # 绘制文本线条粗细
        #tf = max(tl - 1, 1)
        cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
        cv2.imwrite(save_path_img, image)

        logger.info(f'Saving image to: {save_path_img}')

    except Exception as e:
        logger.error('Failed to draw bounding box and label on image.')
        logger.error(e)

    print('Done. (%.3fs)' % (time.time() - t0))

    # 输出日志信息
    if matched_name != 'Unknown':
        logger.info('Matching successful, Distance = %.2f, Distance_percent = %s, Name = %s' % (
            min_distance, distance_percent, matched_name))
    else:
        logger.info('Matching failed, no facial information available.')
