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

import yaml

from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler


class ModelLoader:
    _instance = None

    def __new__(cls, model_path):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model_path = model_path
            cls._instance.load_models()
        return cls._instance

    def load_models(self):
        with open('/data/ai/faceSDK/config/model_conf.yaml') as f:
            model_conf = yaml.load(f, Loader=yaml.FullLoader)

        # Load face detection model
        scene = 'non-mask'
        model_category = 'face_detection'
        model_name = model_conf[scene][model_category]
        try:
            faceDetModelLoader = FaceDetModelLoader(self.model_path, model_category, model_name)
            model, cfg = faceDetModelLoader.load_model()
            self.faceDetModelHandler = FaceDetModelHandler(model, 'cuda:0', cfg)
            print('Success loading face detection model.')
        except Exception as e:
            print('Failed to load face detection model.')
            print(e)
            sys.exit(-1)

        # Load face alignment model
        model_category = 'face_alignment'
        model_name = model_conf[scene][model_category]
        try:
            faceAlignModelLoader = FaceAlignModelLoader(self.model_path, model_category, model_name)
            model, cfg = faceAlignModelLoader.load_model()
            self.faceAlignModelHandler = FaceAlignModelHandler(model, 'cuda:0', cfg)
            print('Success loading face alignment model.')
        except Exception as e:
            print('Failed to load face alignment model.')
            print(e)
            sys.exit(-1)

        # Load face recognition model
        model_category = 'face_recognition'
        model_name = model_conf[scene][model_category]
        try:
            faceRecModelLoader = FaceRecModelLoader(self.model_path, model_category, model_name)
            model, cfg = faceRecModelLoader.load_model()
            self.faceRecModelHandler = FaceRecModelHandler(model, 'cuda:0', cfg)
            print('Success loading face recognition model.')
        except Exception as e:
            print('Failed to load face recognition model.')
            print(e)
            sys.exit(-1)

    def get_face_det_model_handler(self):
        return self.faceDetModelHandler

    def get_face_align_model_handler(self):
        return self.faceAlignModelHandler

    def get_face_rec_model_handler(self):
        return self.faceRecModelHandler
