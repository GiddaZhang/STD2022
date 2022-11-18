dependencies = ['torch', 'numpy', 'resampy', 'soundfile']

from .model.vggish import VGGish

model_urls = {
    'vggish': 'https://github.com/harritaylor/torchvggish/'
              'releases/download/v0.1/vggish-10086976.pth',
    'pca': 'https://github.com/harritaylor/torchvggish/'
           'releases/download/v0.1/vggish_pca_params-970ea276.pth'
}
model_ckpt = {
    'vggish': './torchvggish/vggish-10086976.pth',
    'pca': './torchvggish/vggish_pca_params-970ea276.pth'
}

def vggish(**kwargs):
    model = VGGish(ckpt=model_ckpt, **kwargs)
    return model
