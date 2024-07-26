import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from tqdm import tqdm

from config import Config
from data import get_training_data, get_validation_data
from loss import ColorLoss
from models import *
from utils import seed_everything, save_checkpoint

opt = Config('training.yml')

seed_everything(opt.OPTIM.SEED)

if not os.path.exists(opt.TRAINING.SAVE_DIR):
    os.makedirs(opt.TRAINING.SAVE_DIR)


def train():
    # Accelerate
    accelerator = Accelerator(log_with='wandb') if opt.OPTIM.WANDB else Accelerator()

    config = {
        "dataset": opt.TRAINING.TRAIN_DIR,
        "model": opt.MODEL.SESSION
    }
    accelerator.init_trackers("film", config=config)
    metric_color = ColorLoss()
    loss_mse = torch.nn.MSELoss()

    # Data Loader
    train_dir = opt.TRAINING.TRAIN_DIR
    val_dir = opt.TRAINING.VAL_DIR

    train_dataset = get_training_data(train_dir, opt.MODEL.FILM, img_options={'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    trainloader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16, drop_last=False, pin_memory=True)
    val_dataset = get_validation_data(val_dir, opt.MODEL.FILM, img_options={'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    testloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)

    # Model
    model = FilmNet()

    # Optimizer & Scheduler
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.OPTIM.LR_INITIAL,
                            betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS, eta_min=opt.OPTIM.LR_MIN)

    trainloader, testloader = accelerator.prepare(trainloader, testloader)
    model = accelerator.prepare(model)
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

    start_epoch = 1
    best_psnr = 0

    size = len(testloader)

    # training
    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        model.train()

        for _, data in enumerate(tqdm(trainloader)):
            inp = data[0].contiguous()
            tar = data[1]

            # forward
            optimizer.zero_grad()
            res = model(inp)

            train_loss = loss_mse(inp, res) + 0.4 * (1 - structural_similarity_index_measure(inp, res))

            # backward
            accelerator.backward(train_loss)
            optimizer.step()

        scheduler.step()

        # testing
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model.eval()
            with torch.no_grad():
                psnr = 0
                ssim = 0
                delta_e = 0
                for _, test_data in enumerate(tqdm(testloader)):
                    inp = test_data[0].contiguous()
                    tar = test_data[1]

                    res = model(inp, tar)
                    all_res, all_tar = accelerator.gather((res, tar))
                    psnr += peak_signal_noise_ratio(all_res, all_tar)
                    ssim += structural_similarity_index_measure(all_res, all_tar)
                    delta_e += metric_color(all_res, all_tar)

                psnr /= size
                ssim /= size
                delta_e /= size

                if psnr > best_psnr:
                    # save model
                    best_psnr = psnr
                    save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, epoch, opt.TRAINING.SAVE_DIR)

                accelerator.log({
                    "PSNR": psnr,
                    "SSIM": ssim,
                    "ΔE": delta_e
                }, step=epoch)

                print(
                    "epoch: {}, PSNR: {}, SSIM: {}, ΔE: {}, best PSNR: {}".format(epoch, psnr, ssim, delta_e,
                                                                                  best_psnr))

    accelerator.end_training()


if __name__ == '__main__':
    train()
