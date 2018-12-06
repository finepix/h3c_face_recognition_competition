#coding=utf-8
import cv2 as cv
import time
import os
import cv2
import numpy as np
from collections import Counter
import time

FILE_SUFFIX = '.jpg'
NUMPY_SAVE_PATH = 'p2_video_id_name'


class Extractor():
    '''
        extract video to split by frame
    '''
    # def __init__(self, video_path, save_dir):
    #     self.video_path = video_path
    #     self.file_name = os.path.basename(video_path)
    #     self.save_dir = save_dir

    def __init__(self, model, f, video_dir):
        self.model = model
        self.f = f
        self.video_dir = video_dir

    def video_exatrc(self):
        '''

        :return: tmp image dir for video(used for further learning)
        '''
        vc = cv.VideoCapture(self.video_path)  # 读入视频文件

        if vc.isOpened():  # 判断是否正常打开
            rval, frame = vc.read()
        else:
            rval = False

        timeF = 1  # 视频帧计数间隔频率
        c = 1
        file_dir = os.path.join(os.path.dirname(self.save_dir), self.file_name.split('.')[0])
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        while rval:  # 循环读取视频帧
            rval, frame = vc.read()
            if frame is None:
                break
            if (c % timeF == 0):  # 每隔timeF帧进行存储操作
                file_path =  file_dir + '/' + str(c) + FILE_SUFFIX
                cv.imwrite(file_path, frame)  # 存储为图像

            c = c + 1
            cv.waitKey(1)
        vc.release()
        print("video convert to pictures successfully!")

        return file_dir

    def video_recognition(self, feas, ids):
        rs = []
        drop_lists = []
        # todo
        files = os.listdir(self.video_dir)
        # files.sort(lambda x: int(x.split('_')[-1].split('.')[0]))
        files.sort()
        count = 1
        for file in files:
            if file == '..' or file == '.':
                continue

            if file.split('.')[1] != 'mp4' and file.split('.')[1] != 'avi':
                continue

            video_id = file.split('.')[0]

            print('recognize video:', video_id)
            print('index:', count)
            count += 1

            if count > 100:
                # todo
                break

            file = os.path.join(self.video_dir, file)

            cap = cv2.VideoCapture(file)
            for i in range(1, 10000):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 5 * i)  # 设置要获取的帧号
                a, b = cap.read()  # read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
                if not a:
                    break

                # todo
                cv2.imshow('b', b)
                cv2.waitKey(1)

                (result_list, drop_list) = self.f(b, self.model, feas, ids)
                result_set = list(set(result_list))

                drop_lists += drop_list


                for r in result_set:
                    rs.append((video_id, r))

                # cv2.imshow('b', b)
                # cv2.waitKey(1)


            # drop list
            if len(drop_lists) > 0:
                cb = Counter(drop_lists).most_common(1)
                if len(cb) < 1:
                    continue
                rs.append((video_id, cb[0]).decode('utf-8'))
                print('################call back:' + cb[0]).decode('utf-8')

            time.sleep(2)

        np.savez(NUMPY_SAVE_PATH, rs)
        print('np save rs in video')

        return rs
