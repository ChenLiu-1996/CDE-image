import argparse
import os
from typing import Tuple
from glob import glob

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch_ema import ExponentialMovingAverage
from torch.utils.data import Dataset
from tqdm import tqdm
import albumentations as A

from util.prepare_dataset import prepare_dataset
from util.attribute_hashmap import AttributeHashmap
from util.metrics import psnr, ssim, dice_coeff
from util.seed import seed_everything

from nn.scheduler import LinearWarmupCosineAnnealingLR
from nn.unet_cde_simple import CDEUNet


def add_random_noise(img: torch.Tensor, max_intensity: float = 0.1) -> torch.Tensor:
    intensity = max_intensity * torch.rand(1).to(img.device)
    noise = intensity * torch.randn_like(img)
    return img + noise

def train(config: AttributeHashmap):
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')

    train_set, val_set, test_set, num_image_channel = \
        prepare_dataset(config=config)

    # Build the model
    model = CDEUNet(device=device,
                    num_filters=config.num_filters,
                    depth=config.depth,
                    in_channels=num_image_channel,
                    out_channels=num_image_channel)

    ema = ExponentialMovingAverage(model.parameters(), decay=0.9)

    model.to(device)
    model.init_params()
    ema.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                              warmup_epochs=config.max_epochs//10,
                                              max_epochs=config.max_epochs)

    mse_loss = torch.nn.MSELoss()
    best_val_psnr = 0
    backprop_freq = config.batch_size
    if 'n_plot_per_epoch' not in config.keys():
        config.n_plot_per_epoch = 1

    os.makedirs(config.save_folder + 'train/', exist_ok=True)
    os.makedirs(config.save_folder + 'val/', exist_ok=True)

    recon_psnr_thr = 20
    recon_good_enough = False

    for epoch_idx in tqdm(range(config.max_epochs)):
        model, ema, optimizer, scheduler = \
            train_epoch(config=config, device=device, train_set=train_set, model=model,
                        epoch_idx=epoch_idx, ema=ema, optimizer=optimizer, scheduler=scheduler,
                        mse_loss=mse_loss, backprop_freq=backprop_freq, train_time_dependent=recon_good_enough)

        with ema.average_parameters():
            model.eval()
            val_recon_psnr, val_pred_psnr = \
                val_epoch(config=config, device=device, val_set=val_set, model=model, epoch_idx=epoch_idx)

            if val_recon_psnr > recon_psnr_thr:
                recon_good_enough = True

            if val_pred_psnr > best_val_psnr:
                best_val_psnr = val_pred_psnr
                model.save_weights(config.model_save_path.replace('.pty', '_best_pred_psnr.pty'))

    return


def train_epoch(config: AttributeHashmap,
                device: torch.device,
                train_set: Dataset,
                model: torch.nn.Module,
                epoch_idx: int,
                ema: ExponentialMovingAverage,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler._LRScheduler,
                mse_loss: torch.nn.Module,
                backprop_freq: int,
                train_time_dependent: bool):
    '''
    Training epoch for many models.
    '''

    train_loss_recon, train_loss_pred, train_recon_psnr, train_recon_ssim, train_pred_psnr, train_pred_ssim = 0, 0, 0, 0, 0, 0
    model.train()
    optimizer.zero_grad()

    plot_freq = int(len(train_set) // config.n_plot_per_epoch)
    for iter_idx, (images, timestamps) in enumerate(tqdm(train_set)):

        if 'max_training_samples' in config:
            if iter_idx > config.max_training_samples:
                break

        shall_plot = iter_idx % plot_freq == 0

        # NOTE: batch size is set to 1,
        # because in Neural ODE, `eval_times` has to be a 1-D Tensor,
        # while different patients have different `timestamps` in our dataset.
        # We will simulate a bigger batch size when we handle optimizer update.

        # images: [1, N, C, H, W], containing [x_start, ..., x_end]
        # timestamps: [1, N], containing [t_start, ..., t_end]
        assert images.shape[1] >= 2
        assert timestamps.shape[1] >= 2

        x_list, t_arr = convert_variables(images, timestamps, device)
        x_noisy_list = [add_random_noise(x) for x in x_list]

        x_end_pred = model(x=torch.vstack(x_noisy_list[:-1]), t=(t_arr - t_arr[0]) * config.t_multiplier)
        import pdb
        pdb.set_trace()

        ################### Recon Loss to update Encoder/Decoder ##################
        # Unfreeze the model.
        model.unfreeze()

        x_start_recon = model(x=x_start_noisy, t=torch.zeros(1).to(device))
        x_end_recon = model(x=x_end_noisy, t=torch.zeros(1).to(device))

        loss_recon = mse_loss(x_start, x_start_recon) + mse_loss(x_end, x_end_recon)
        train_loss_recon += loss_recon.item()

        # Simulate `config.batch_size` by batched optimizer update.
        loss_recon = loss_recon / backprop_freq
        loss_recon.backward()

        ################## Pred Loss to update time-dependent modules #############
        # Freeze all time-independent modules.
        try:
            model.freeze_time_independent()
        except AttributeError:
            print('`model.freeze_time_independent()` ignored.')

        if train_time_dependent:
            assert torch.diff(t_list).item() > 0
            x_end_pred = model(x=x_start_noisy, t=torch.diff(t_list) * config.t_multiplier)
            loss_pred = mse_loss(x_end, x_end_pred)
            train_loss_pred += loss_pred.item()

            # Simulate `config.batch_size` by batched optimizer update.
            loss_pred = loss_pred / backprop_freq
            loss_pred.backward()

        else:
            # Will not train the time-dependent modules until the reconstruction is good enough.
            with torch.no_grad():
                x_end_pred = model(x=x_start_noisy, t=torch.diff(t_list) * config.t_multiplier)
                loss_pred = mse_loss(x_end, x_end_pred)
                train_loss_pred += loss_pred.item()

        # Simulate `config.batch_size` by batched optimizer update.
        if iter_idx % config.batch_size == config.batch_size - 1:
            optimizer.step()
            optimizer.zero_grad()
            ema.update()

        x0_true, x0_recon, xT_true, xT_recon, xT_pred = \
            numpy_variables(x_start, x_start_recon, x_end, x_end_recon, x_end_pred)

        # NOTE: Convert to image with normal dynamic range.
        x0_true, x0_recon, xT_true, xT_recon, xT_pred = \
            cast_to_0to1(x0_true, x0_recon, xT_true, xT_recon, xT_pred)

        train_recon_psnr += psnr(x0_true, x0_recon) / 2 + psnr(xT_true, xT_recon) / 2
        train_recon_ssim += ssim(x0_true, x0_recon) / 2 + ssim(xT_true, xT_recon) / 2
        train_pred_psnr += psnr(xT_true, xT_pred)
        train_pred_ssim += ssim(xT_true, xT_pred)

        if shall_plot:
            save_path_fig_sbs = '%s/train/figure_log_epoch%s_sample%s.png' % (
                config.save_folder, str(epoch_idx + 1).zfill(5), str(iter_idx + 1).zfill(5))
            plot_side_by_side(t_list, x0_true, xT_true, x0_recon, xT_recon, xT_pred, save_path_fig_sbs)

    train_loss_pred, train_loss_recon, train_recon_psnr, train_recon_ssim, train_pred_psnr, train_pred_ssim = \
        [item / len(train_set.dataset) for item in (train_loss_pred, train_loss_recon, train_recon_psnr, train_recon_ssim, train_pred_psnr, train_pred_ssim)]

    scheduler.step()

    log('Train [%s/%s] loss [recon: %.3f, pred: %.3f], PSNR (recon): %.3f, SSIM (recon): %.3f, PSNR (pred): %.3f, SSIM (pred): %.3f'
        % (epoch_idx + 1, config.max_epochs, train_loss_recon, train_loss_pred, train_recon_psnr,
            train_recon_ssim, train_pred_psnr, train_pred_ssim),
        filepath=config.log_dir,
        to_console=False)

    return model, ema, optimizer, scheduler


@torch.no_grad()
def val_epoch(config: AttributeHashmap,
              device: torch.device,
              val_set: Dataset,
              model: torch.nn.Module,
              epoch_idx: int):
    val_recon_psnr, val_recon_ssim, val_pred_psnr, val_pred_ssim = 0, 0, 0, 0

    plot_freq = int(len(val_set) // config.n_plot_per_epoch)
    for iter_idx, (images, timestamps) in enumerate(tqdm(val_set)):
        shall_plot = iter_idx % plot_freq == 0

        assert images.shape[1] == 2
        assert timestamps.shape[1] == 2

        # images: [1, 2, C, H, W], containing [x_start, x_end]
        # timestamps: [1, 2], containing [t_start, t_end]
        x_start, x_end, t_list = convert_variables(images, timestamps, device)

        x_start_recon = model(x=x_start, t=torch.zeros(1).to(device))
        x_end_recon = model(x=x_end, t=torch.zeros(1).to(device))
        x_end_pred = model(x=x_start, t=torch.diff(t_list) * config.t_multiplier)

        x0_true, x0_recon, xT_true, xT_recon, xT_pred = \
            numpy_variables(x_start, x_start_recon, x_end, x_end_recon, x_end_pred)

        # NOTE: Convert to image with normal dynamic range.
        x0_true, x0_recon, xT_true, xT_recon, xT_pred = \
            cast_to_0to1(x0_true, x0_recon, xT_true, xT_recon, xT_pred)

        val_recon_psnr += psnr(x0_true, x0_recon) / 2 + psnr(xT_true, xT_recon) / 2
        val_recon_ssim += ssim(x0_true, x0_recon) / 2 + ssim(xT_true, xT_recon) / 2
        val_pred_psnr += psnr(xT_true, xT_pred)
        val_pred_ssim += ssim(xT_true, xT_pred)

    val_recon_psnr, val_recon_ssim, val_pred_psnr, val_pred_ssim = \
        [item / len(val_set.dataset) for item in (
            val_recon_psnr, val_recon_ssim, val_pred_psnr, val_pred_ssim)]

    return val_recon_psnr, val_pred_psnr


def convert_variables(images: torch.Tensor,
                      timestamps: torch.Tensor,
                      device: torch.device) -> Tuple[torch.Tensor]:
    '''
    Some repetitive processing of variables.
    '''
    x_list = [images[:, i, ...].float().to(device) for i in range(images.shape[1])]
    t_arr = timestamps[0].float().to(device)
    return x_list, t_arr


def numpy_variables(*tensors: torch.Tensor) -> Tuple[np.array]:
    '''
    Some repetitive numpy casting of variables.
    '''
    return [_tensor.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0) for _tensor in tensors]

def cast_to_0to1(*np_arrays: np.array) -> Tuple[np.array]:
    '''
    Cast image to normal dynamic range between 0 and 1.
    '''
    return [np.clip((_arr + 1) / 2, 0, 1) for _arr in np_arrays]

def gray_to_rgb(*tensors: torch.Tensor) -> Tuple[np.array]:
    rgb_list = []
    for item in tensors:
        assert len(item.shape) in [2, 3]
        if len(item.shape) == 3:
            assert item.shape[-1] == 1
            rgb_list.append(np.repeat(item, 3, axis=-1))
        else:
            rgb_list.append(np.repeat(item[..., None], 3, axis=-1))

    return rgb_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entry point.')
    parser.add_argument('--gpu-id', help='Index of GPU device', default=0)
    parser.add_argument('--run-count', default=None, type=int)
    parser.add_argument('--dataset-path', default=os.path.abspath('../data/synthesized/'), type=str)
    parser.add_argument('--image-folder', default='base', type=str)
    parser.add_argument('--output-save-path', default=os.path.abspath('../results/synthesized/base/'), type=str)
    parser.add_argument('--learning-rate', default=1e-4, type=float)
    parser.add_argument('--max-epochs', default=100, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--num-filters', default=16, type=int)
    parser.add_argument('--depth', default=5, type=int)
    parser.add_argument('--t-multiplier', default=0.1, type=float)
    parser.add_argument('--num-workers', default=0, type=int)
    parser.add_argument('--random-seed', default=1, type=int)
    parser.add_argument('--train-val-test-ratio', default='6:2:2', type=str)
    parser.add_argument('--max-training-samples', default=1000, type=int)
    args = vars(parser.parse_args())
    config = AttributeHashmap(args)

    # Initialize save folder.
    if config.run_count is None:
        existing_runs = glob(config.output_save_path + '/run_*/')
        if len(existing_runs) > 0:
            run_counts = [int(item.split('/')[-2].split('run_')[1]) for item in existing_runs]
            run_count = max(run_counts) + 1
        else:
            run_count = 1
    config.save_folder = '%s/run_%d/' % (config.output_save_path, run_count)
    config.model_save_path = config.save_folder + config.output_save_path.split('/')[-2] + '.pty'

    seed_everything(config.random_seed)
    train(config=config)
