# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import mdnc
import pytorch_lightning as pl
import coloredlogs, logging
coloredlogs.install()
logger = logging.getLogger(__name__)
from torch import optim



class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        # lr_factor = self.warmup_steps ** 0.5 * min(epoch ** (-0.5), epoch * self.warmup_steps ** (-1.5))

        return lr_factor


class Regressor(pl.LightningModule):
    def __init__(self, param_model, random_state=0, pos_weight=None):
        super().__init__()
        # log hyperparameters
        self.param_model = param_model
        self.save_hyperparameters()

        # loss function
        self.criterion = nn.MSELoss()

    # =============================================================================
    # train / val / test
    # =============================================================================
    def forward(self, x):
        x = self.model(x)
        return x

    def _shared_step(self, batch):
        x_ppg, y, x_abp, peakmask, vlymask = batch
        pred, hidden = self.model(x_ppg)
        loss = self.criterion(pred, x_abp)
        return loss, pred, x_abp, y

    def training_step(self, batch, batch_idx):
        loss, pred_abp, t_abp, label = self._shared_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return {"loss": loss, "pred_abp": pred_abp, "true_abp": t_abp, "true_bp": label}

    def training_epoch_end(self, train_step_outputs):
        logit = torch.cat([v["pred_abp"] for v in train_step_outputs], dim=0)
        label = torch.cat([v["true_abp"] for v in train_step_outputs], dim=0)
        metrics = self._cal_metric(logit.detach(), label.detach())
        self._log_metric(metrics, mode="train")

    def validation_step(self, batch, batch_idx):
        loss, pred_abp, t_abp, label = self._shared_step(batch)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return {"loss": loss, "pred_abp": pred_abp, "true_abp": t_abp, "true_bp": label}

    def validation_epoch_end(self, val_step_end_out):
        logit = torch.cat([v["pred_abp"] for v in val_step_end_out], dim=0)
        label = torch.cat([v["true_abp"] for v in val_step_end_out], dim=0)
        metrics = self._cal_metric(logit.detach(), label.detach())
        self._log_metric(metrics, mode="val")
        return val_step_end_out

    def test_step(self, batch, batch_idx):
        loss, pred_abp, t_abp, label = self._shared_step(batch)
        self.log('test_loss', loss, prog_bar=True)
        return {"loss": loss, "pred_abp": pred_abp, "true_abp": t_abp, "true_bp": label}

    def test_epoch_end(self, test_step_end_out):
        logit = torch.cat([v["pred_abp"] for v in test_step_end_out], dim=0)
        label = torch.cat([v["true_abp"] for v in test_step_end_out], dim=0)
        metrics = self._cal_metric(logit.detach(), label.detach())
        self._log_metric(metrics, mode="test")
        return test_step_end_out

    def _cal_metric(self, logit: torch.tensor, label: torch.tensor):
        mse = torch.mean((logit - label) ** 2)
        mae = torch.mean(torch.abs(logit - label))
        me = torch.mean(logit - label)
        std = torch.std(torch.mean(logit - label, dim=1))
        return {"mse": mse, "mae": mae, "std": std, "me": me}

    def _log_metric(self, metrics, mode):
        for k, v in metrics.items():
            self.log(f"{mode}_{k}", v, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            # self.log(f"{mode}_{k}", v, on_step=False, on_epoch=True)

    # =============================================================================
    # optimizer
    # =============================================================================
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.param_model.lr)
        if self.param_model.get("scheduler_WarmUp"):
            logger.info("!!!!!!!! is using warm up !!!!!!!!")
            self.lr_scheduler = {"scheduler": CosineWarmupScheduler(optimizer, **(self.param_model.scheduler_WarmUp)),
                                 "monitor": "val_loss"}
            return [optimizer], self.lr_scheduler
        return optimizer


# %%
class Unet1d(Regressor):
    def __init__(self, param_model, random_state=0):
        super(Unet1d, self).__init__(param_model, random_state)

        self.model = mdnc.modules.conv.UNet1d(param_model.output_channel,
                                              list(param_model.layers),
                                              in_planes=param_model.input_size,
                                              out_planes=1)

    def _shared_step(self, batch):
        x, y, x_abp, peakmask, vlymask = batch
        pred = self.model(x['ppg'])
        loss = self.criterion(pred, x_abp)
        return loss, pred, x_abp, y, peakmask, vlymask

    def training_step(self, batch, batch_idx):
        loss, pred_abp, t_abp, label, peakmask, vlymask = self._shared_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return {"loss": loss, "pred_abp": pred_abp, "true_abp": t_abp, "true_bp": label, "mask_pk": peakmask,
                "mask_vly": vlymask}

    def training_epoch_end(self, train_step_outputs):
        logit = torch.cat([v["pred_abp"] for v in train_step_outputs], dim=0)
        label = torch.cat([v["true_abp"] for v in train_step_outputs], dim=0)
        mask_pk = torch.cat([v["mask_pk"] for v in train_step_outputs], dim=0)
        mask_vly = torch.cat([v["mask_vly"] for v in train_step_outputs], dim=0)

        metrics = self._cal_metric(logit.detach(), label.detach())
        self._log_metric(metrics, mode="train")

    def validation_step(self, batch, batch_idx):
        loss, pred_abp, t_abp, label, peakmask, vlymask = self._shared_step(batch)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return {"loss": loss, "pred_abp": pred_abp, "true_abp": t_abp, "true_bp": label, "mask_pk": peakmask,
                "mask_vly": vlymask}

    def validation_epoch_end(self, val_step_end_out):
        logit = torch.cat([v["pred_abp"] for v in val_step_end_out], dim=0)
        label = torch.cat([v["true_abp"] for v in val_step_end_out], dim=0)
        mask_pk = torch.cat([v["mask_pk"] for v in val_step_end_out], dim=0)
        mask_vly = torch.cat([v["mask_vly"] for v in val_step_end_out], dim=0)

        metrics = self._cal_metric(logit.detach(), label.detach())
        self._log_metric(metrics, mode="val")
        return val_step_end_out

    def test_step(self, batch, batch_idx):
        loss, pred_abp, t_abp, label, peakmask, vlymask = self._shared_step(batch)
        self.log('test_loss', loss, prog_bar=True)
        return {"loss": loss, "pred_abp": pred_abp, "true_abp": t_abp, "true_bp": label, "mask_pk": peakmask,
                "mask_vly": vlymask}

    def test_epoch_end(self, test_step_end_out):
        logit = torch.cat([v["pred_abp"] for v in test_step_end_out], dim=0)
        label = torch.cat([v["true_abp"] for v in test_step_end_out], dim=0)
        mask_pk = torch.cat([v["mask_pk"] for v in test_step_end_out], dim=0)
        mask_vly = torch.cat([v["mask_vly"] for v in test_step_end_out], dim=0)

        metrics = self._cal_metric(logit.detach(), label.detach())
        self._log_metric(metrics, mode="test")
        return test_step_end_out

    def _cal_metric(self, logit: torch.tensor, label: torch.tensor):
        mse = torch.mean((logit - label) ** 2)
        mae = torch.mean(torch.abs(logit - label))
        me = torch.mean(logit - label)
        std = torch.std(torch.mean(logit - label, dim=1))
        # true_pk = torch.masked_select(label, mask_pk)
        # pred_pk = torch.masked_select(logit, mask_pk)
        # print(len(mask_pk), print(label.shape))

        # true_vly = torch.masked_select(label, mask_vly)
        # pred_vly = torch.masked_select(logit, mask_vly)

        # mse = 0.5*(torch.mean((pred_pk-true_pk)**2) + torch.mean((pred_vly-true_vly)**2))
        # mae = 0.5*(torch.mean(torch.abs(pred_pk-true_pk)) + torch.mean(torch.abs(pred_vly-true_vly)))
        # std = 0.5*(torch.std(torch.mean(pred_pk-true_pk, dim=1)) + torch.std(torch.mean(pred_vly-true_vly, dim=1)))
        # me = 0.5*(torch.mean(pred_pk-true_pk) + torch.mean(pred_vly-true_vly))
        return {"mse": mse, "mae": mae, "std": std, "me": me}

    # %%


if __name__ == '__main__':
    from omegaconf import OmegaConf
    import pandas as pd
    import numpy as np
    import joblib
    import os

    os.chdir('/sensorsbp/code/train')
    from core.loaders.wav_loader import WavDataModule
    from core.utils import get_nested_fold_idx, cal_statistics
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.callbacks import LearningRateMonitor
    from core.models.trainer import MyTrainer

    config = OmegaConf.load('/sensorsbp/code/train/core/config/hydra/unet_sensors_valstd.yaml')
    all_split_df = joblib.load(config.exp.subject_dict)
    config = cal_statistics(config, all_split_df)
    for foldIdx, (folds_train, folds_val, folds_test) in enumerate(get_nested_fold_idx(5)):
        if foldIdx == 0:  break
    train_df = pd.concat(np.array(all_split_df)[folds_train])
    val_df = pd.concat(np.array(all_split_df)[folds_val])
    test_df = pd.concat(np.array(all_split_df)[folds_test])

    dm = WavDataModule(config)
    dm.setup_kfold(train_df, val_df, test_df)
    # dm.train_dataloader()
    # dm.val_dataloader()
    # dm.test_dataloader()

    # init model
    model = Unet1d(config.param_model)
    early_stop_callback = EarlyStopping(**dict(config.param_early_stop))
    checkpoint_callback = ModelCheckpoint(**dict(config.logger.param_ckpt))
    lr_logger = LearningRateMonitor()

    trainer = MyTrainer(**dict(config.param_trainer), callbacks=[early_stop_callback, checkpoint_callback, lr_logger])

    # trainer main loop
    trainer.fit(model, dm)