import numpy as np
import wave
import math
import matplotlib.pyplot as plt
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from torchvggish.hubconf import vggish
def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):#如果是文件夹那么递归调用一下
            del_file(c_path)
        else:                    #如果是一个文件那么直接删除
            os.remove(c_path)


audios_dir = 'Test/Noise/audio'
save_dir = 'Test/Test_Noise/audio'
save_feat_dir = 'Test/Test_Noise/afeat'


vggish = vggish(pretrained=True)
vggish.eval()

def nextpow2(n):
    return np.ceil(np.log2(np.abs(n))).astype('long')
def berouti(SNR):
    if -5.0 <= SNR <= 20.0:
        a = 4 - SNR * 3 / 20
    else:
        if SNR < -5.0:
            a = 5
        if SNR > 20:
            a = 1
    return a
def berouti1(SNR):
    if -5.0 <= SNR <= 20.0:
        a = 3 - SNR * 2 / 20
    else:
        if SNR < -5.0:
            a = 4
        if SNR > 20:
            a = 1
    return a
def find_index(x_list):
    index_list = []
    for i in range(len(x_list)):
        if x_list[i] < 0:
            index_list.append(i)
    return index_list
def audio_denoise(input_audio):
    f = wave.open(input_audio)
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    fs = framerate
    str_data = f.readframes(nframes)
    f.close()
    x = np.frombuffer(str_data, dtype=np.short)
    len_ = 20 * fs // 1000      # 帧长度
    PERC = 50                   # 窗口重叠占帧的百分比
    len1 = len_ * PERC // 100   # 窗口重叠部分
    len2 = len_ - len1          # 窗口非重叠部分
    Thres = 3                   # 噪声功率收敛阈值
    Expnt = 2.0                 # Expnt=1:幅度谱；Expnt=2:功率谱；
    beta = 0.002                # 谱下限参数
    G = 0.9
    # 初始化汉明窗
    win = np.hamming(len_)
    winGain = len2 / sum(win)

    # 噪声幅度计算
    nFFT = 2 * 2 ** (nextpow2(len_))
    noise_mean = np.zeros(nFFT)

    j = 0
    # 前五帧，噪声频谱幅度|D(w)|的均值
    for k in range(1, 6):
        ####################
        noise_mean = noise_mean + np.abs(np.fft.fft(win * x[j:j + len_], n=nFFT))
        ####################
        j = j + len_
        continue

    noise_mu = noise_mean / 5
    # 分配内存并初始化各种变量
    k = 1
    img = 1j
    x_old = np.zeros(len1)
    Nframes = len(x) // len2 - 1
    xfinal = np.zeros(Nframes * len2)
    for n in range(0, Nframes):
        # 窗口
        insign = win * x[k - 1:k + len_ - 1]
        # FFT
        spec = np.fft.fft(insign, nFFT)
        # 计算幅值
        sig = np.abs(spec)
        # 存储相位信息
        theta = np.angle(spec)

        # 信噪比计算
        SNRseg = 0
        ####################
        info_power = np.sum(sig**2)
        noise_power = np.sum(noise_mu**2)
        SNRseg = 10 * math.log10(info_power / noise_power)
        ####################
        

        if Expnt == 1.0:
            alpha = berouti1(SNRseg)
        else:
            alpha = berouti(SNRseg)

        # 谱减
        
        # |X(w)|^Expnt = |Y(w)|^Expnt - alpha*|D(w)|^Expnt, when > 0,
        # |X(w)|^Expnt = beta*|D(w)|^Expnt, else.

        sub_speech = np.zeros(theta.shape)
        ####################
        sub_speech = sig ** Expnt - alpha * noise_mu ** Expnt
        ####################


        # 当纯净信号小于噪声信号时的处理
        diffw = sub_speech - beta * noise_mu ** Expnt
        z = find_index(diffw)       # 寻找小于0的项
        if len(z) > 0:
            sub_speech[z] = beta * noise_mu[z] ** Expnt
        
        # 更新噪声频谱
        if SNRseg < Thres:  
            noise_temp = G * noise_mu ** Expnt + (1 - G) * sig ** Expnt
            noise_mu = noise_temp ** (1 / Expnt)

        # 信号恢复
        sub_speech[nFFT // 2 + 1:nFFT] = np.flipud(sub_speech[1:nFFT // 2])
        x_phase = np.zeros(theta.shape)
        ####################
        x_phase = sub_speech ** (1 / Expnt) * np.exp(img * theta)
        ####################
        xi = np.fft.ifft(x_phase).real

        # 重叠叠加
        xfinal[k-1:k + len2 - 1] = x_old + xi[0:len1]
        x_old = xi[0 + len1:len_]
        k = k + len2

        #进度条，可删
        print("\r", end="")
        print("进度: {}%: ".format(n * 100 // (Nframes-1)), "▓" * (n * 100 // (Nframes-1)), end="")
        sys.stdout.flush()
    # 保存结果
    wf = wave.open(os.path.join(save_dir,input_audio[-8:]), 'wb')
    wf.setparams(params)
    wave_data = (winGain * xfinal).astype(np.short)
    wf.writeframes(wave_data.tobytes())
    wf.close()

    sname = input_audio[-8:-4] + '.npy'
    feat = vggish.forward(os.path.join(save_dir,input_audio[-8:]))
    print(feat.shape, sname)
    np.save(os.path.join(save_feat_dir, sname), feat.detach().cpu().numpy())
    print(str(input_audio[-8:])+' finished.')
    del_file(save_dir)

def all_denoise(audios_dir):
    anames = os.listdir(audios_dir)
    for aname in anames:
        audio_denoise(os.path.join(audios_dir,aname))

all_denoise(audios_dir)