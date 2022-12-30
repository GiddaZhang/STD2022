import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import urllib
import os
import scipy.ndimage  # For image filtering
import imageio  # For loading images
import numpy as np
from numpy import array, zeros, arange, exp, random, ones_like, zeros_like, ones
from scipy.fftpack import dct
from scipy.fftpack import idct
from scipy.signal import convolve, convolve2d, correlate2d, correlate
from imageio import imread
import itertools
import math
import matplotlib.pyplot as plt
import unittest
from PIL import Image
import cv2
import os
import sys
from torchvision.models import resnet101, ResNet101_Weights
import torch

model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
model.cuda()
model.eval()


def extract_feature(model, x):
    mu = torch.from_numpy(np.array([0.485, 0.456, 0.406])).float().unsqueeze(0).unsqueeze(2).unsqueeze(
        3).cuda()
    std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).float().unsqueeze(0).unsqueeze(2).unsqueeze(
        3).cuda()
    x = (x - mu) / std
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return x


def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):  # 如果是文件夹那么递归调用一下
            del_file(c_path)
        else:  # 如果是一个文件那么直接删除
            os.remove(c_path)


videos_dir = 'Test/Noise/video'
save_dir = 'Test/Test_Noise/video'
save_feat_dir = 'Test/Test_Noise/vfeat'


def video_denoise_extract(model, video_dir):
    #TODO:将一个视频去噪后提取特征保存到相应文件夹中
    #Input:
    # video_dir:待去噪视频的相对路径名;
    # save_dir:保存视频的文件夹的相对路径名，但是会随时删除;
    # save_feat_dir:保存特征的文件夹的相对路径名
    #Output:None

    #以下为去噪部分
    cap = cv2.VideoCapture(video_dir)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的高度
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))  # 视频的编码
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 设置输出视频为mp4格式
    cap_fps = fps
    size = (width, height)
    file_list = os.listdir
    video = cv2.VideoWriter(os.path.join(
        save_dir, video_dir[-8:]), fourcc, cap_fps, size)  # 视频格式
    i = 0
    timeF = int(fps)  # 视频帧计数间隔频率
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break

        denoised_frame = cv2.fastNlMeansDenoisingColored(
            frame, None, 10, 10, 7, 15)  # 去噪
        denoised_frame.dtype = np.uint8
        video.write(denoised_frame)  # 生成视频
        cv2.waitKey(1)  # 延时1ms
        i += 1
        print("\r", end="")
        print("进度: {}%: ".format(i * 100 // ((timeF+1)*10)),
              "▓" * (i * 100 // ((timeF+1)*10*2)), end="")
        sys.stdout.flush()
    cap.release()  # 释放视频对象
    video.release()

    #以下为提取特征值部分
    sname = video_dir[-8:-4] + '.npy'
    cap = cv2.VideoCapture(os.path.join(save_dir, video_dir[-8:]))
    cnt = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        if cnt == 0:
            h = frame.shape[0]
            w = frame.shape[1]
            if h > w:
                news = (int(w / h * 224), 224)
            else:
                news = (224, int(h / w * 224))
        cnt += 1
    cap.release()
    step = int(cnt / 10)
    x = np.zeros((10, 224, 224, 3))
    cap = cv2.VideoCapture(os.path.join(save_dir, video_dir[-8:]))
    cnt = 0
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        if cnt == step * i:
            if h > w:
                x[i, :, int((224 - news[0]) / 2):int((224 - news[0]) / 2) +
                  news[0]] = cv2.resize(frame, news)[:, :, ::-1].copy()
            else:
                x[i, int((224 - news[1]) / 2):int((224 - news[1]) / 2) +
                  news[1], :] = cv2.resize(frame, news)[:, :, ::-1].copy()
            i += 1
        if i == 10:
            break
        cnt += 1

    cap.release()
    x = torch.from_numpy(x / 255.0).float().cuda().permute(0, 3, 1, 2)
    with torch.no_grad():
        feat = extract_feature(model, x).cpu().numpy()
    print(feat.shape, sname)
    np.save(os.path.join(save_feat_dir, sname), feat)
    del_file(save_dir)
    print(str(video_dir[-8:]) + ' finished.')


def all_denoise_extract(model, videos_dir):
    # os.system('mkdir {}/vfeat'.format(save_dir))
    vnames = os.listdir(videos_dir)
    list_tmp = [(x[0:4], x) for x in vnames]
    list_tmp.sort()

    for vname in list_tmp:
      video_denoise_extract(model, os.path.join(videos_dir, vname[1]))


all_denoise_extract(model, videos_dir)
