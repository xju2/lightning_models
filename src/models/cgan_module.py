from typing import Any, List

import torch
import torch.nn.functional as F

from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from scipy import stats

class CGANModule(LightningModule):
    def __init__(
        self,
        noise_dim: int,
        num_max_hadrons: int,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        optimizer_generator: torch.optim.Optimizer,
        optimizer_discriminator: torch.optim.Optimizer,
        scheduler_generator: torch.optim.lr_scheduler,
        scheduler_discriminator: torch.optim.lr_scheduler,
    ):
        super().__init__()
        
        self.save_hyperparameters(
            logger=False, ignore=["generator", "discriminator"])
        
        self.generator = generator
        self.discriminator = discriminator
        
        ## loss function
        self.criterion = torch.nn.BCELoss()
        
        ## metric objects for calculating and averaging accuracy across batches
        self.train_loss_gen = MeanMetric()
        self.train_loss_disc = MeanMetric()
        self.val_wd = MeanMetric()
        self.val_nll = MeanMetric()
        
        # for tracking best so far
        self.val_wd_best = MaxMetric()
        self.val_nll_best = MaxMetric()
        
    def forward(self, x: torch.Tensor):
        return self.generator(x)
    
    
    def generate_noise(self, num_evts: int):
        return torch.randn(num_evts, self.hparams.noise_dim)
    
    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_wd_best.reset()
        self.val_nll_best.reset()
        
    def training_step(self, batch: Any, batch_idx: int, optimizer_idx: int):
        real_label = 1
        fake_label = 0
        
        cond_info, x_truth = batch
        
        num_evts = x_truth.shape[0]
        noise = self.generate_noise(num_evts)
        
        label = torch.full((num_evts,), real_label, dtype=torch.float)
        ## Train generator
        if optimizer_idx == 0:
            x_fake = torch.concat([cond_info, noise], dim=1)
            fake = self.generator(x_fake)
            x_generated = torch.cat([cond_info, fake], dim=1)
            score_fakes = self.discriminator(self.generator(x_generated))
            loss_gen = self.criterion(score_fakes, label)
            
            ## update and log metrics
            self.train_loss_gen(loss_gen)
            self.log("lossG", loss_gen, prog_bar=True)
            
            return {"lossG": loss_gen}
        
        ##  Train discriminator   
        if optimizer_idx == 1:
            ## with real batch  
            score_truth = self.discriminator(x_truth).squeeze()
            loss_real = self.criterion(score_truth, label)

            ## with fake batch
            x_fake = torch.concat([cond_info, noise], dim=1)
            fake = self.generator(x_fake)
            x_generated = torch.cat([cond_info, fake], dim=1)
            
            score_fakes = self.discriminator(x_generated.detach()).squeeze()
            label.fill_(fake_label)
            loss_fake = self.criterion(score_fakes, label)
            
            loss_disc = (loss_real + loss_fake) / 2
            
            ## update and log metrics
            self.train_loss_disc(loss_disc)
            self.log("lossD", loss_disc, prog_bar=True)
            return {"lossD": loss_disc}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass    
        
    def validation_step(self, batch: Any, batch_idx: int):
        cond_info, x_truth = batch
        num_samples, num_dims = x_truth.shape
        cond_dim = cond_info.shape[1]
        output_dims = num_dims - cond_dim
        noise = self.generate_noise(num_samples)
        x_input = torch.concat([cond_info, noise], dim=1)
        
        samples = self.generator(x_input)
        
        ## evaluate the accuracy of hadron types
        ## with likelihood ratio    
        gen_types = samples[:, 2:].reshape(num_samples* self.hparams.num_max_hadrons, -1)
        log_probability = F.log_softmax(gen_types, dim=1)
        loss_types = float(F.nll_loss(log_probability, x_truth[:, cond_dim+2:]))
        
        ## compute the WD for the first two angle variables
        ## eta and phi.
        samples = samples.cpu().detach().numpy()
        
        distances = [
            stats.wasserstein_distance(samples[:, idx], x_truth[:, cond_dim+idx]) \
                for idx in range(2)
        ]
        wd_distance = sum(distances)/len(distances)
        
        ## update and log metrics
        self.val_wd(wd_distance)
        self.val_nll(loss_types)
        self.log("val_wd", wd_distance, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_nll", loss_types, on_step=False, on_epoch=True, prog_bar=True)
        
        return {"wd": wd_distance, "nll": loss_types, "preds": samples}
        
        
    def validaton_epoch_end(self, outputs: List[Any]):
        pass
    
    def configure_optimizers(self):
        return [self.optimizer_generator, self.optimizer_discriminator], \
               [self.scheduler_generator, self.scheduler_discriminator]