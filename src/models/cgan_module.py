from typing import Any, List, Optional

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from scipy import stats
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class CondParticleGANModule(LightningModule):
    """Conditional GAN predicting particle momenta and types"""
    def __init__(
        self,
        noise_dim: int,
        num_particle_ids: int,  ## maximum number of particle types
        num_output_hadrons: int,      ## number of outgoing hadrons
        num_particle_kinematics: int, ## number of kinematic variables
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        optimizer_generator: torch.optim.Optimizer,
        optimizer_discriminator: torch.optim.Optimizer
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
        
        cond_info, x_momenta, x_type_indices = batch
        num_evts = cond_info.shape[0]
        device = x_momenta.device
        

        noise = self.generate_noise(num_evts).to(device)
        label = torch.full((num_evts,), real_label, dtype=torch.float).to(device)
        ## Train generator
        if optimizer_idx == 0:
            
            x_fake = noise if cond_info is None else torch.concat([cond_info, noise], dim=1)
            fakes = self.generator(x_fake)
            
            # particle_kinematics = fakes[:, :-self.hparams.num_particle_ids]
            # particle_types = fakes[:, -self.hparams.num_particle_ids:].reshape(
            #     num_evts* self.hparams.num_particle_ids, -1)
            # log_probability = F.log_softmax(particle_types, dim=1)
            x_generated = fakes if cond_info is None else torch.cat([cond_info, fakes], dim=1)
            score_fakes = self.discriminator(x_generated).squeeze()
            loss_gen = self.criterion(score_fakes, label)
            
            ## update and log metrics
            self.train_loss_gen(loss_gen)
            self.log("lossG", loss_gen, prog_bar=True)
            
            return {"loss": loss_gen}
        
        ##  Train discriminator   
        if optimizer_idx == 1:
            ## with real batch
            x_truth = x_momenta if cond_info is None else torch.cat([cond_info, x_momenta], dim=1)
            if x_type_indices is not None:
                x_types = F.one_hot(x_type_indices.reshape(-1), num_classes=self.hparams.num_particle_ids).reshape(
                    num_evts, -1)
                x_truth = torch.cat([x_truth, x_types], dim=1)
                
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
            return {"loss": loss_disc}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass    
        
    def validation_step(self, batch: Any, batch_idx: int):
        cond_info, x_momenta, x_type_indices = batch
        num_evts, _ = x_momenta.shape
        
        ## generate events from the Generator
        x_input = self.generate_noise(num_evts).to(x_momenta.device)
        if cond_info is not None:
            x_input = torch.concat([cond_info, x_input], dim=1)
        samples = self.generator(x_input)

        avg_nll = 0
        if x_type_indices is not None:
            
            ## evaluate the accuracy of hadron types
            ## with likelihood ratio
            for pidx in range(self.hparams.num_output_hadrons):
                gen_types = samples[:, self.hparams.num_particle_kinematics+pidx*self.hparams.num_particle_ids:self.hparams.num_particle_kinematics+(pidx+1)*self.hparams.num_particle_ids]
                log_probability = F.log_softmax(gen_types, dim=1)
                nll = float(F.nll_loss(log_probability,
                                    x_type_indices[:, pidx]))
                avg_nll += nll
                
            avg_nll = avg_nll / self.hparams.num_output_hadrons
        
        ## compute the WD for the particle kinmatics
        samples = samples.cpu().detach().numpy()
        x_momenta = x_momenta.cpu().detach().numpy()
        
        distances = [
            stats.wasserstein_distance(samples[:, idx], x_momenta[:, idx]) \
                for idx in range(self.hparams.num_particle_kinematics)
        ]
        wd_distance = sum(distances)/len(distances)
        
        ## update and log metrics
        self.val_wd(wd_distance)
        self.val_nll(avg_nll)
        self.log("val_wd", wd_distance, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_nll", avg_nll, on_step=False, on_epoch=True, prog_bar=True)
        
        return {"wd": wd_distance, "nll": avg_nll, "preds": samples}
        
        
    def validaton_epoch_end(self, outputs: List[Any]):
        pass
    
    def configure_optimizers(self):
        opt_gen = self.hparams.optimizer_generator(params=self.generator.parameters())
        opt_disc = self.hparams.optimizer_discriminator(params=self.discriminator.parameters())
        
        return [opt_gen, opt_disc], []