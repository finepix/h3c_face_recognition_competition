import face_model
import argparse
import cv2
import sys
import numpy as np
import shutil
import os
from video_extractor import Extractor


# model path
FEATURE_MODELR100_PATH = '/home/tt/test/models/feature_extraction_model/model-r100-ii/model'
FEATURE_MODELR34_PATH = '/home/tt/test/models/feature_extraction_model/model-r34-amf/model'
MTCNN_MODEL_PATH = '/home/tt/test/models/detection_and_alignment_model/mtcnn-model'

# default face dir and feature data dir
DEFAULT_FACE_PATH = '/home/tt/test/data/image/'
DEFAULT_FEATURE_PATH = '/home/tt/test/data/features/fea.bin'
DEFAULT_IDENTITIES_PATH = '/home/tt/test/data/features/identities.bin'
DEFAULT_IDENTITIES_PATH_1 = '/home/tt/test/data/features/identities.bin.npz'
UNREADABLE_IMAGE_DIR = '/home/tt/test/data/unuse_img/unreadable'
MORE_FACES_IMG_DIR = '/home/tt/test/data/unuse_img/more'
P1_RS_PATH_1 = '/home/tt/test/result/p1_1.txt'
P1_RS_PATH_2 = '/home/tt/test/result/p1_2.txt'
P2_RS_PATH = '/home/tt/test/result/p2.txt'



# video path
VIDEO_PATH = '/home/tt/test/data/video/515.mp4'
VIDEO_DIR = '/home/tt/data/VIDEOS/diertishiping'
TMP_DIR = '/home/tt/test/data/tmp/'

# images dir
# IMAGES_DIR_1 = '/home/tt/test/data/p1_img/no1'
# IMAGES_DIR_2 = '/home/tt/test/data/p1_img/no2'
IMAGES_DIR_1 = '/home/chuangke9/chuangke/diyi/no1'
IMAGES_DIR_2 = '/home/chuangke9/chuangke/diyi/no2'


def get_features_from_file(feature_file_path=DEFAULT_FEATURE_PATH, ids_file_path=DEFAULT_IDENTITIES_PATH_1):
    """

    :param feature_file_path:
    :param ids_file_path:
    :return:
    """
    feas = np.fromfile(feature_file_path, dtype=np.float64)
    feas.shape = -1, 512
    print(feas.shape)
    ids = np.load(ids_file_path)['arr_0']
    return feas, ids


def calculate_sim_1_by_N(x, y, metric='cosin'):
    '''
        define deffrent metric for similarity, and calculate the similarity of x and y
    :param x:
    :param y: feartures from labeled image
    :return: cosin similarity or euclidean dist
    '''
    if metric == 'cosin':
        # cosin similarity
        return np.dot(x, y.T)
    elif metric == 'euclidean':
        # if x is 1 by n
        x = np.expand_dims(x, axis=0)
        num_test = y.shape[0]
        num_faces = x.shape[0]
        dists =np.zeros((num_test,num_faces))
        dists = np.sqrt(
            -2 * np.dot(y, x.T) + np.sum(np.square(x), axis=1) + np.transpose([np.sum(np.square(y), axis=1)]))
        return dists
    else:
        print('err')
        return 5


def p2_recognize_function(img, model, feas, ids):
    rs = []
    drop_list = []
    faces = model.get_inputs(img)
    # if not found face in the images, continue to calculate the next image
    if faces is None:
        print('face not found in file.')
        return rs, drop_list
    # iter the faces in the image, and calculate the similarity
    for face in faces:

        fea = model.get_feature(face)

        metric = 'cosin'
        if metric == 'euclidean':
            sim = calculate_sim_1_by_N(fea, feas, metric=metric)
            index = np.where(sim == min(sim))[0]

            if len(index) > 1:
                index = index[0]

            if (sim[index] < 1):
                # print('{}  confident:{}' % (identities[int(index)], str(sim[index])))
                # TypeError: list indices must be integers, not tuple
                print(ids[int(index)])
                # todo
                rs.append(ids[int(index)])
            else:
                print('greatter than 1, dist:' + str(sim[index]))
                print(ids[int(index)] + ' drop')
                # todo
                drop_list.append(ids[int(index)])

                continue
        else:
            sim = calculate_sim_1_by_N(fea, feas, metric=metric)
            index = np.where(sim == max(sim))[0]

            if len(index) > 1:
                index = index[0]

            if (sim[index] > 0.4):
                print(ids[int(index)] + ' similarity:' + str(sim[index]))
                # todo
                rs.append(ids[int(index)])
            else:
                print('lower than 0.4, confident:' + str(sim[index]))
                print(ids[int(index)] + ' drop')
                # todo
                if sim[index] > 0.3:
                    drop_list.append(ids[int(index)])
                continue
    return (rs, drop_list)


def video_recognize_frame_by_frame(model,proccess_function, feas, ids, video_dir=VIDEO_DIR):
    """

    :param model:
    :param proccess_function:
    :param video_dir:
    :return:
    """
    extractor = Extractor(model, proccess_function, video_dir)
    # todo
    rs = extractor.video_recognition(feas, ids)
    return rs


def video_frame_face_recognition(model,face_features, identities, video_path=VIDEO_PATH, tmp_save_dir=TMP_DIR):
    """
                if the max similarity are bigger than the threshhold return the indexed name of the person, otherwise
            return null
    :param model:
    :param face_features:
    :param identities:
    :param video_path:
    :param tmp_save_dir:
    :return:
    """
    extractor = Extractor(video_path, tmp_save_dir)
    tmp_image_dir = extractor.video_exatrc()

    # list all files in dir , and sorted by the nums
    image_files = os.listdir(tmp_image_dir)
    image_files.sort(key=lambda x: int(x.split('.')[0]))

    for image_file in image_files:
        file = os.path.join(tmp_image_dir, image_file)

        print('recognize file:' + image_file)
        img = cv2.imread(file)

        faces = model.get_inputs(img)
        # if not found face in the images, continue to calculate the next image
        if faces is None:
            print('face not found in file:' + file)
            continue
        # iter the faces in the image, and calculate the similarity
        if len(faces) > 1:
            #todo
            print('test')

        for face in faces:

            tmp = np.transpose(face, (1, 2, 0))
            # view face by cv2
            cv2.imshow('Face', tmp)
            print('face in ' + image_file)
            cv2.waitKey(2)

            fea = model.get_feature(face)

            metric = 'euclidean'
            if metric == 'euclidean':
                sim = calculate_sim_1_by_N(fea, face_features, metric=metric)
                index = np.where(sim == min(sim))[0]
                # todo
                if (sim[index] < 1.24):
                    # print('{}  confident:{}' % (identities[int(index)], str(sim[index])))
                    # TypeError: list indices must be integers, not tuple
                    print(identities[int(index)] + '  dist:' + str(sim[index]))
                else:
                    print('greatter than 1.24, dist:' + str(sim[index]))
                    continue
            else:
                sim = calculate_sim_1_by_N(fea, face_features, metric=metric)
                index = np.where(sim == max(sim))[0]
                # todo
                if (sim[index] > 0.3):
                    print(identities[int(index)] + '  confident:' + str(sim[index]))
                else:
                    print('lower than 0.3, confident:' + str(sim[index]))
                    continue

    return identities


def p1_write_to_file(rs, p1_rs_path=P1_RS_PATH_1):
    """
                names, image_id
    :param rs:
    :param p1_rs_path:
    :return:
    """
    # file = open(p1_rs_path)
    # if not os.path.exists(file):

    with open(p1_rs_path, 'w') as f:
        f.write('name' + '\t' + 'image_id' + '\n')

        for r in rs:
            line = r[0] + '\t' + r[1] + '\n'

            print line

            f.write(line)
    print('result save to file:', p1_rs_path)


def p2_write_to_file(rs, p2_rs_path=P2_RS_PATH):
    with open(p2_rs_path, 'w') as f:
        f.write('video_id' + '\t' + 'name' + '\n')

        for r in rs:
            line = r[0] + '\t' + r[1] + '\n'

            print line

            f.write(line)
    print('result save to file:', p2_rs_path)


def test_write_file(feas, ids):
    """
        test write to file
    :param feas:
    :param ids:
    :return:
    """
    rs = []
    for i in range(0, 100):
        rs.append((ids[i], 'file name'))
    p1_write_to_file(rs)



def test():
    # args initial
    parser = argparse.ArgumentParser(description='video test')
    # general
    parser.add_argument('--image-size',
                        default='112,112')
    parser.add_argument('--model',
                        default=FEATURE_MODELR100_PATH)
    parser.add_argument('--mtcnn_path',
                        default=MTCNN_MODEL_PATH)
    parser.add_argument('--gpu',
                        default=0, type=int)
    parser.add_argument('--det',
                        default=0, type=int)
    parser.add_argument('--flip',
                        default=0, type=int)
    parser.add_argument('--threshold',
                        default=1.24, type=float)
    args = parser.parse_args()

    # model load
    model = face_model.FaceModel(args)

    # feature extraction
    # face_path = DEFAULT_FACE_PATH
    # print('###########start to embed features.###############')
    # # feas, ids = get_features_from_raw_image(model, face_path, True)
    # feas, ids = get_features_from_raw_image_not_delete(model, face_path, True)
    # print('feature embedding completed.')



    # # extract features from video and calculate the similarity
    # print('##########recognize people in videos.#############')
    # video_frame_face_recognition(model, human_face_features, identities)

    feas, ids = get_features_from_file()
    print(len(ids))
    print(feas.shape)


    # todo
    # test n1 folder
    # rs = p1_recognize_images_files(model, feas, ids, images_dir=IMAGES_DIR_1)
    # p1_write_to_file(rs, P1_RS_PATH_1)
    #
    # # test n2 folder
    # rs = p1_recognize_images_files(model, feas, ids, images_dir=IMAGES_DIR_2)
    # p1_write_to_file(rs, P1_RS_PATH_2)

    # test
    # test_write_file(feas, ids)

    # todo
    rs = video_recognize_frame_by_frame(model, p2_recognize_function, feas, ids)
    p2_write_to_file(rs)
    print('video recognition completed.')




if __name__ == '__main__':
    test()
