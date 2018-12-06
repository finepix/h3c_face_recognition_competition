#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:f-zx 
@file: test_1.py 
@time: 2018/10/27 
"""
import cv2
import os

cap = cv2.VideoCapture('v.mp4')  # 返回一个capture对象
# cap.set(cv2.CAP_PROP_POS_FRAMES, 50)  # 设置要获取的帧号
# a, b = cap.read()  # read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
# cv2.imshow('b', b)
# cv2.waitKey(1000)

dir_ = '.'

files = os.listdir(dir_)
for file in files:
    if file.split('.')[1] != 'mp4':
        continue
    cap = cv2.VideoCapture(file)
    for i in range(1, 1000):
        cap.set (cv2.CAP_PROP_POS_FRAMES, 5 * i)  # 设置要获取的帧号
        a, b = cap.read ()  # read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
        if not a:
            break
        cv2.imshow ('b', b)
        cv2.waitKey (0)