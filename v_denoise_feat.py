import os
from torchvision.models import resnet101, ResNet101_Weights
import cv2
import torch
import numpy as np
import sys

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


def extract_dir(model, dirname, save_dir):
    #os.system('mkdir {}/vfeat'.format(dirname))
    vnames = os.listdir(os.path.join(dirname, 'video'))
    # 排序
    list_tmp = [(x[0:4], x) for x in vnames]
    list_tmp.sort()
    for vname_ in list_tmp:
        vname = vname_[1]
        sname = vname[:-4] + '.npy'
        x = np.zeros((10, 224, 224, 3))
        cap = cv2.VideoCapture(os.path.join(dirname, 'video', vname))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的宽度
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的高度
        step = int(cap.get(cv2.CAP_PROP_FPS))  # 获取视频的帧率
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))  # 视频的编码
        if h > w:
            news = (int(w / h * 224), 224)
        else:
            news = (224, int(h / w * 224))
        cnt = 0
        i = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            if cnt == step * i:
                frame = cv2.fastNlMeansDenoisingColored(
                    frame, None, 10, 10, 7, 15)  # 去噪
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
            print("\r", end="")
            print("进度: {}%: ".format((i + 1) * 10),
                  "=" * ((i + 1) * 10), end="")
            sys.stdout.flush()

        cap.release()
        x = torch.from_numpy(x / 255.0).float().cuda().permute(0, 3, 1, 2)
        with torch.no_grad():
            feat = extract_feature(model, x).cpu().numpy()
        print(feat.shape, sname)
        np.save(os.path.join(save_dir, 'vfeat', sname), feat)


video_dir = 'Test/Noise'
save_dir = 'Test/Test_Noise'

extract_dir(model, video_dir, save_dir)
