import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device

from midas.model_loader import default_models, load_model

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    from model.diffusion_model import Unet, GaussianDiffusion

    do_epipolar = config['args']['do_epipolar']
    do_bidirectional_epipolar = config['args']['do_bidirectional_epipolar']
    do_blur = config['args']['do_blur']
    do_mae = config['args']['do_mae']
    do_latent = config['args']['do_latent']
    big_unet = config['args']['big_unet']
    mask_ratio = config['args']['mask_ratio']

    if big_unet:
        #* 使用 512x512 image 做midas 
        model_type = "dpt_beit_large_512"
    else:
        model_type = "dpt_swin2_tiny_256"

    model_weights = default_models[model_type]
    midas_model, midas_transform, net_w, net_h = load_model("cpu", model_weights, model_type, False, None, False)

    # setup data_loader instances
    train_data_loader = config.init_obj('data_loader', module_data, mode="train",
                                        midas_transform = midas_transform,
                                        do_latent = do_latent,
                                        )
    valid_data_loader = None

    if big_unet:
        dimmults = (1,2,4,8)
    else:
        dimmults = (2,4,8)

    if do_latent:
        #* latent shape 為 (4,64,64)
        unet_model = Unet(
            dim = 128,
            init_dim = 128,
            dim_mults = dimmults,
            channels=4, 
            out_dim=4,
            do_epipolar = do_epipolar,
            do_bidirectional_epipolar = do_bidirectional_epipolar,
            do_blur = do_blur,
            do_mae = do_mae,
            mask_ratio = mask_ratio,
        )
      
    else:
      unet_model = Unet(
          dim = 128,
          init_dim = 128,
          dim_mults = dimmults,
          channels=3, 
          out_dim=3,
          do_epipolar = do_epipolar,
          do_bidirectional_epipolar = do_bidirectional_epipolar,
          do_blur = do_blur,
          do_mae = do_mae,
          mask_ratio = mask_ratio,
      )
    
    model = GaussianDiffusion(
        unet_model,
        timesteps = 1000,    # number of steps
        sampling_timesteps = 50,  # ddim sample
        beta_schedule = 'cosine',
        do_latent = do_latent,
    )

    # logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    midas_model = midas_model.to(device)

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      midas_model = midas_model,
                      )

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
