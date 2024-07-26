import os
import warnings
warnings.filterwarnings('ignore')

import torch
from accelerate import Accelerator
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from loss import ColorLoss
from tqdm import tqdm

from config import Config
from data import get_validation_data
from models import *
from utils import seed_everything, load_checkpoint

opt = Config('training.yml')

seed_everything(opt.OPTIM.SEED)


def test():
    accelerator = Accelerator()

    model = FilmNet()
    load_checkpoint(model, opt.TESTING.WEIGHT)

    # Data Loader
    val_dir = opt.TRAINING.VAL_DIR
    val_dataset = get_validation_data(val_dir, opt.MODEL.FILM, img_options={'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    testloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)

    metric_color = ColorLoss()

    model, testloader = accelerator.prepare(model, testloader)

    model.eval()

    with torch.no_grad():
        delta_e = 0
        stat_ssim = 0
        stat_psnr = 0
        size = len(testloader)
        for idx, test_data in enumerate(tqdm(testloader)):
            # get the inputs; data is a list of [targets, inputs, filename]
            inp = test_data[0]
            tar = test_data[1]

            res = model(inp)
            save_image(res, os.path.join(os.getcwd(), "result", test_data[2][0] + '_pred.png'))
            save_image(tar, os.path.join(os.getcwd(), "result", test_data[2][0] + '_gt.png'))

            stat_psnr += peak_signal_noise_ratio(res, tar, data_range=1)
            stat_ssim += structural_similarity_index_measure(res, tar, data_range=1)
            delta_e += metric_color(res, tar)

        delta_e /= size
        stat_ssim /= size
        stat_psnr /= size

    print("PSNR: {}, SSIM: {}, ΔE: {}".format(stat_psnr, stat_ssim, delta_e))


if __name__ == '__main__':
    test()
