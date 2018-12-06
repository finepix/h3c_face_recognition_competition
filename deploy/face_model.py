from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
#import tensorflow as tf
import numpy as np
import mxnet as mx
import random
import cv2
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
from mtcnn_detector import MtcnnDetector
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'common'))
import face_image
import face_preprocess


def do_flip(data):
  for idx in xrange(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

def get_model(ctx, image_size, model_str, layer, epoch=0):

  print('loading',model_str, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(model_str, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  return model

class FaceModel:
  def __init__(self, args):
    self.args = args
    ctx = mx.gpu(args.gpu)
    _vec = args.image_size.split(',')
    assert len(_vec)==2
    image_size = (int(_vec[0]), int(_vec[1]))
    self.model = None
    if len(args.model)>0:
      self.model = get_model(ctx, image_size, args.model, 'fc1')

    self.threshold = args.threshold
    self.det_minsize = 50
    self.det_threshold = [0.6,0.7,0.8]
    #self.det_factor = 0.9
    self.image_size = image_size
    mtcnn_path = args.mtcnn_path
    if args.det==0:
      detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=self.det_threshold)
    else:
      detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=[0.0,0.0,0.2])
    self.detector = detector

  def get_inputs_for_feature_extract(self, face_img):
    '''
          if img have more than one face ,drop
    :param face_img:
    :return:
    '''
    if face_img is None:
      print('img None.')
      return None
    ret = self.detector.detect_face(face_img, det_type=self.args.det)

    if ret is None:
      return None
    bbox, points = ret

    if bbox.shape[0] == 0:
      return None

    if bbox.shape[0] > 1 and bbox[1, 4] > 0.99:
      return None

    if bbox[0, 4] < 0.9:
      return None

    # select the first face of bbox
    print(bbox.shape)
    print(bbox[:, 4])
    box = bbox[0, 0:4]
    point = points[0, :].reshape((2, 5)).T

    nimg = face_preprocess.preprocess(face_img, box, point, image_size='112,112')
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)

    #cv2.imshow('Face', nimg)
    #print('face confident:' + str(bbox[0, 4]))
    #cv2.waitKey(2)

    aligned = np.transpose(nimg, (2, 0, 1))

    # cv2.destroyAllWindows()
    return aligned


  def get_inputs(self, face_img):
    '''
        return several faces in the image
    :param face_img:
    :return:
    '''
    ret = self.detector.detect_face(face_img, det_type = self.args.det)
    if ret is None:
      return None
    bbox, points = ret
    if bbox.shape[0]==0:
      return None

    # iter faces
    print(bbox.shape)
    iters = bbox.shape[0]
    rs = []

    for i in range(iters):

      box = bbox[i, 0:4]
      point = points[i, :].reshape((2, 5)).T

      # print(box)
      # print(point)
      if bbox[i, 4] < 0.99:
        print('face confident low.')
        continue

      nimg = face_preprocess.preprocess(face_img, box, point, image_size='112,112')
      nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)

      cv2.imshow('Face', nimg)
      print('face confident:' + str(bbox[i, 4]))
      cv2.waitKey(2)

      aligned = np.transpose(nimg, (2, 0, 1))

      rs.append(aligned)

    # cv2.destroyAllWindows()
    return rs



  def get_input(self, face_img):
    '''
        get one face in face_image, select the rank1
    :param face_img:
    :return:
    '''
    if face_img is None:
      print('img None.')
      return None
    ret = self.detector.detect_face(face_img, det_type = self.args.det)

    if ret is None:
      return None
    bbox, points = ret

    if bbox.shape[0]==0:
      return None

    # if bbox.shape[0] > 1:
    #   return None

    # select the first face of bbox
    print(bbox.shape)
    print(bbox[:, 4])
    box = bbox[0, 0:4]
    point = points[0, :].reshape((2, 5)).T

    nimg = face_preprocess.preprocess(face_img, box, point, image_size='112,112')
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)

    # cv2.imshow('Face', nimg)
    # print('face confident:' + str(bbox[0, 4]))
    # cv2.waitKey(2)

    aligned = np.transpose(nimg, (2, 0, 1))

    # cv2.destroyAllWindows()
    return aligned


  def get_feature(self, aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    self.model.forward(db, is_train=False)
    embedding = self.model.get_outputs()[0].asnumpy()
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding

