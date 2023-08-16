import os
import warnings

import cv2
import numpy as np
import torch
from accelerate import Accelerator
from skimage import io, color
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from config import Config
from data import get_validation_data
from models import *
from utils import seed_everything, load_checkpoint

warnings.filterwarnings('ignore')

opt = Config('training.yml')

seed_everything(opt.OPTIM.SEED)


def test():
    accelerator = Accelerator()
    device = accelerator.device

    # Data Loader
    val_dir = opt.TRAINING.VAL_DIR

    val_dataset = get_validation_data(val_dir, opt.MODEL.FILM, {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    testloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False,
                            pin_memory=True)

    if opt.MODEL.SESSION == 'LUT':
        model = FilmNet()
    else:
        params = {'batch_norm': False}
        model = Enhancer(params=params, device=device, ll_layer=opt.MODEL.LL, enhance=opt.MODEL.ENHANCE)

    load_checkpoint(model, opt.TESTING.WEIGHT)

    model, testloader = accelerator.prepare(model, testloader)

    model.eval()

    with torch.no_grad():
        delta_e = 0
        stat_ssim = 0
        stat_psnr = 0
        size = len(testloader)
        for idx, test_data in enumerate(tqdm(testloader)):
            # get the inputs; data is a list of [targets, inputs, filename]
            tar = test_data[0]
            inp = test_data[1]

            res = model(inp, tar)
            save_image(res, os.path.join(os.getcwd(), "result", test_data[2][0] + '_pred.png'))
            save_image(tar, os.path.join(os.getcwd(), "result", test_data[2][0] + '_gt.png'))

            pred_img = cv2.imread(os.path.join(os.getcwd(), "result", test_data[2][0] + '_pred.png'))
            gt_img = cv2.imread(os.path.join(os.getcwd(), "result", test_data[2][0] + '_gt.png'))

            stat_psnr += psnr(pred_img, gt_img, data_range=255)
            stat_ssim += ssim(pred_img, gt_img, data_range=255, multichannel=True)
            delta_e += compare_images(
                os.path.join(os.getcwd(), "result", test_data[2][0] + '_pred.png'),
                os.path.join(os.getcwd(), "result", test_data[2][0] + '_gt.png')
            )

        delta_e /= size
        stat_ssim /= size
        stat_psnr /= size

    print("PSNR: {}, SSIM: {}, ΔE: {}".format(stat_psnr, stat_ssim, delta_e))


def compare_images(str_img_orig, str_img_edit):
    # read images
    im_orig = io.imread(str_img_orig)
    im_edit = io.imread(str_img_edit)

    # convert to lab
    lab_orig = color.rgb2lab(im_orig)
    lab_edit = color.rgb2lab(im_edit)

    # calculate difference
    de_diff = color.deltaE_cie76(lab_orig, lab_edit)

    return np.mean(de_diff)


if __name__ == '__main__':
    test()
