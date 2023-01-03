import models
import torch
import numpy as np
import os


def gen_tsample(n, test_num):
    tsample = np.zeros((500, n)).astype(np.int16)
    for i in range(500):
        tsample[i] = np.random.permutation(test_num)[:n]
    # tsample: (500, n)
    np.save('tsample_{}.npy'.format(n), tsample)

def get_top(tsample, rst):
    top1 = 0.0
    top5 = 0.0
    n = tsample.shape[1]
    for i in range(500):
        idx = tsample[i]
        rsti = rst[idx][:,idx]
        assert rsti.shape[0] == n
        assert rsti.shape[1] == n
        sorti = np.sort(rsti, axis=1)
        for j in range(n):
            if rsti[j,j] == sorti[j,-1]:
                top1 += 1
            if rsti[j,j] >= sorti[j,-5]:
                top5 += 1
    top1 = top1 / 500 / n
    top5 = top5 / 500 / n
    print('Top1 accuracy for sample {} is: {}.'.format(n, top1))
    print('Top5 accuracy for sample {} is: {}.'.format(n, top5))

def main():
    epoch = 100
    model = models.FrameByFrame()
    ckpt = torch.load('checkpoints/noise/VA_METRIC_state_epoch{}.pth'.format(epoch), map_location='cpu')
    model.load_state_dict(ckpt)
    model.cuda().eval()

    vpath = 'Test/Denoise/vfeat'
    apath = 'Test/Denoise/afeat'
    test_num = 804

    rst = np.zeros((test_num, test_num))
    vfeats = torch.zeros(test_num, 512, 10).float()
    afeats = torch.zeros(test_num, 128, 10).float()
    for i in range(test_num):
        vfeat = np.load(os.path.join(vpath, '%04d.npy'%i))
        for j in range(test_num):
            vfeats[j] = torch.from_numpy(vfeat).float().permute(1, 0)
            afeat = np.load(os.path.join(apath, '%04d.npy'%j))
            afeats[j] = torch.from_numpy(afeat).float().permute(1, 0)
        with torch.no_grad():
            out = model(vfeats.cuda(), afeats.cuda())
        rst[i] = (out[:,1] - out[:,0]).cpu().numpy()
        if i % 10 is 0:
            print(i)

    np.save('clean.npy'.format(epoch), rst)

    # rst = np.load('rst_epoch{}.npy'.format(epoch))
    print('Test checkpoint epoch {}.'.format(epoch))

    # gen_tsample(50, test_num)

    # tsample = np.load('tsample_{}.npy'.format(50))
    # get_top(tsample, rst)

if __name__ == '__main__':
    main()
