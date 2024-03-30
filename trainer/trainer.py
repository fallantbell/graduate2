import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker

import wandb
from tqdm import tqdm

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None,
                 midas_model = None,
                 ):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None

        self.lr_scheduler = lr_scheduler
        self.log_step = int(self.len_epoch//100)

        self.midas_model = midas_model

        # self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        # self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        # self.train_metrics.reset()
        for batch_idx, data in tqdm(enumerate(self.data_loader)):
            #* img shape (b,2,3,h,w)
            #* src_img_tensor 是 (b,3,256,256)
            #* intrinsic 是 (b,4,4) 
            #* c2w shape (b,2,4,4)

            img = data['img'].to(self.device)
            src_img_tensor = data['src_img'].to(self.device)
            intrinsic = data['intrinsics'].to(self.device)
            c2w = data['c2w'].to(self.device)

            self.optimizer.zero_grad()

            with torch.no_grad():
                #* 這裡用 pretrain 的 midas 對 source img encode
                #! 因為未知原因 midas model 無法做平行計算，所以放在外面來做
                src_l2, src_l3,src_l4 = self.midas_model.forward(src_img_tensor)  

            loss = self.model(img, src_l2, src_l3,src_l4, K = intrinsic, c2w = c2w)

            loss = loss.mean()
            loss.backward()
            self.optimizer.step()

            cur_step = (epoch - 1) * self.len_epoch + batch_idx


            if batch_idx % self.log_step == 0:
                
                print('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                
                wandb.log({"epoch":epoch,"loss":loss},step = cur_step)

            if batch_idx == self.len_epoch:
                break

        if self.do_validation:
            val_log = self._valid_epoch(epoch)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return 1

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
