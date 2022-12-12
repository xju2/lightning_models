from typing import Any, List, Optional, Dict, Callable, Tuple

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from scipy import stats
from torchmetrics import MinMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class CondParticleGANModule(LightningModule):
    """Conditional GAN predicting particle momenta and types.
    Parameters:
        noise_dim: dimension of noise vector
        num_particle_ids: maximum number of particle types
        num_output_particles: number of outgoing particles
        num_particle_kinematics: number of outgoing particles' kinematic variables
        generator: generator network
        discriminator: discriminator network
        optimizer_generator: generator optimizer
        optimizer_discriminator: discriminator optimizer
        comparison_fn: function to compare generated and real data
    """
    def __init__(
        self,
        noise_dim: int,
        cond_info_dim: int,
        num_particle_ids: int,
        num_output_particles: int,
        num_particle_kinematics: int,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        optimizer_generator: torch.optim.Optimizer,
        optimizer_discriminator: torch.optim.Optimizer,
        comparison_fn: Optional[Any] = None,
    ):
        super().__init__()
        
        self.save_hyperparameters(
            logger=False, ignore=["generator", "discriminator", "comparison_fn"])
        
        self.generator = generator
        self.discriminator = discriminator
        self.comparison_fn = comparison_fn
        
        ## loss function
        self.criterion = torch.nn.BCELoss()
        
        ## metric objects for calculating and averaging accuracy across batches
        self.train_loss_gen = MeanMetric()
        self.train_loss_disc = MeanMetric()
        self.val_wd = MeanMetric()
        self.val_nll = MeanMetric()
        
        # for tracking best so far
        self.val_min_avg_wd = MinMetric()
        self.val_min_avg_nll = MinMetric()
        
        self.test_wd = MeanMetric()
        self.test_nll = MeanMetric()
        self.test_wd_best = MinMetric()
        self.test_nll_best = MinMetric()
        
    def forward(self, noise: torch.Tensor, cond_info: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x_fake = noise if cond_info is None else torch.concat([cond_info, noise], dim=1)
        fakes = self.generator(x_fake)
        num_evts = noise.shape[0]
        particle_kinematics = fakes[:, :self.hparams.num_particle_kinematics]       # type: ignore
        particle_types = fakes[:, self.hparams.num_particle_kinematics:].reshape(   # type: ignore
            num_evts* self.hparams.num_output_particles, -1)                        # type: ignore
        # particle_type_idx = torch.argmax(particle_types, dim=1).reshape(num_evts, -1)
        return particle_kinematics, particle_types
    
    def configure_optimizers(self):
        opt_gen = self.hparams.optimizer_generator(params=self.generator.parameters()) # type: ignore
        opt_disc = self.hparams.optimizer_discriminator(params=self.discriminator.parameters()) # type: ignore
        
        return [opt_gen, opt_disc], []
    
    def generate_noise(self, num_evts: int):
        return torch.randn(num_evts, self.hparams.noise_dim)    # type: ignore
    
    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_min_avg_wd.reset()
        self.val_min_avg_nll.reset()
        self.test_wd_best.reset()
        self.test_nll_best.reset()
        
    def training_step(self, batch: Any, batch_idx: int, optimizer_idx: int):
        real_label = 1
        fake_label = 0
        
        cond_info, x_momenta, x_type_indices = batch
        num_evts = x_momenta.shape[0]
        device = x_momenta.device
        
        noise = self.generate_noise(num_evts).to(device)
        ## Train generator
        if optimizer_idx == 0:
            particle_kinematics, particle_types = self(noise, cond_info)
            particle_type_idx = torch.argmax(particle_types, dim=1).reshape(num_evts, -1)
            x_generated = particle_kinematics if cond_info is None else torch.cat([cond_info, particle_kinematics], dim=1)
            
            score_fakes = self.discriminator(x_generated, particle_type_idx).squeeze()
            
            label = torch.full((num_evts,), real_label, dtype=torch.float).to(device)
            loss_gen = self.criterion(score_fakes, label)
            
            ## update and log metrics
            self.train_loss_gen(loss_gen)
            self.log("lossG", loss_gen, prog_bar=True)
            
            return {"loss": loss_gen}
        
        ##  Train discriminator   
        if optimizer_idx == 1:
            ## with real batch
            x_truth = x_momenta if cond_info is None else torch.cat([cond_info, x_momenta], dim=1)
            score_truth = self.discriminator(x_truth, x_type_indices).squeeze()
            
            label = torch.full((num_evts,), real_label, dtype=torch.float).to(device)
            loss_real = self.criterion(score_truth, label)

            ## with fake batch
            particle_kinematics, particle_types = self(noise, cond_info)
            particle_type_idx = torch.argmax(particle_types, dim=1).reshape(num_evts, -1)
            x_generated = particle_kinematics if cond_info is None else torch.cat([cond_info, particle_kinematics], dim=1)
            x_generated = x_generated.detach()
            particle_kinematics = particle_kinematics.detach()
            
            score_fakes = self.discriminator(x_generated, particle_type_idx).squeeze()
            fake_labels = torch.full((num_evts,), fake_label, dtype=torch.float).to(device)
            loss_fake = self.criterion(score_fakes, fake_labels)
            
            loss_disc = (loss_real + loss_fake) / 2
            
            ## update and log metrics
            self.train_loss_disc(loss_disc)
            self.log("lossD", loss_disc, prog_bar=True)
            return {"loss": loss_disc}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass    
        
    def step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:    
        """Common steps for valiation and testing"""
        
        cond_info, x_momenta, x_type_indices = batch
        num_evts, _ = x_momenta.shape
        
        ## generate events from the Generator
        noise = self.generate_noise(num_evts).to(x_momenta.device)
        particle_kinematics, particle_types = self(noise, cond_info)
        particle_type_idx = torch.argmax(particle_types, dim=1).reshape(num_evts, -1)
        particle_types = particle_types.reshape(num_evts, -1)
        
        avg_nll = 0
        if x_type_indices is not None:
            ## evaluate the accuracy of hadron types
            ## with likelihood ratio
            for pidx in range(self.hparams.num_output_particles):
                pidx_start = pidx*self.hparams.num_particle_ids
                pidx_end   = pidx_start + self.hparams.num_particle_ids
                gen_types = particle_types[:, pidx_start:pidx_end]
                # print(gen_types.shape, x_type_indices[:, pidx].shape, pidx_start, pidx_end, particle_types.shape)
                log_probability = F.log_softmax(gen_types, dim=1)
                nll = float(F.nll_loss(log_probability,
                                    x_type_indices[:, pidx]))
                avg_nll += nll
                
            avg_nll = avg_nll / self.hparams.num_output_particles
        
        predictions = torch.cat([particle_kinematics, particle_type_idx], dim=1).cpu().detach().numpy()
        truths = torch.cat([x_momenta, x_type_indices], dim=1).cpu().detach().numpy()
        
        ## compute the WD for the particle kinmatics
        x_momenta = x_momenta.cpu().detach().numpy()
        particle_kinematics = particle_kinematics.cpu().detach().numpy()
        distances = [
            stats.wasserstein_distance(particle_kinematics[:, idx], x_momenta[:, idx]) \
                for idx in range(self.hparams.num_particle_kinematics)
        ]
        wd_distance = sum(distances)/len(distances)
        
        return {"wd": wd_distance, "nll": avg_nll, "preds": predictions, "truths": truths}
    

    def compare(self, predictions, truths, outname) -> None:
        """Compare the generated events with the real ones
        Parameters:
            perf: dictionary from the step function
        """
        if self.comparison_fn is not None:
            ## compare the generated events with the real ones
            self.comparison_fn(predictions, truths, outname)
            
            
    def validation_step(self, batch: Any, batch_idx: int):
        """Validation step"""
        perf = self.step(batch, batch_idx)
        wd_distance = perf['wd']
        avg_nll = perf['nll']
        
        ## update and log metrics
        self.val_wd(wd_distance)
        self.val_nll(avg_nll)
        self.log("val_wd", wd_distance, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_nll", avg_nll, on_step=False, on_epoch=True, prog_bar=True)
        
        return perf, batch_idx
        
    def validaton_epoch_end(self, outputs: List[Any]):
        ## `outputs` is a list of dicts returned from `validation_step()`
        
        wd = self.val_wd.compute()
        self.val_min_avg_wd(wd)
        
        # log `val_min_avg_wd` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/min_avg_wd", self.val_min_avg_wd.compute(), prog_bar=True)
        
        ## similiarly for NLL
        nll = self.val_nll.compute()
        self.val_min_avg_nll(nll)
        self.log("val/min_avg_nll", self.val_min_avg_nll.compute(), prog_bar=True)
        
        ## comparison
        print("Best WD: ", self.val_min_avg_wd.compute(), "Best NLL: ", self.val_min_avg_nll.compute())
        print("WD: ", wd, "NLL: ", nll)
        if nll < self.val_min_avg_nll.compute() or \
            wd < self.val_min_avg_wd.compute():
            perf, batch_idx = outputs[0]
            outname = f"val-{self.current_epoch}-{batch_idx}"
            predictions = perf['preds']
            truths = perf['truths']
            self.compare(predictions, truths, outname)
    

    def test_step(self, batch: Any, batch_idx: int):
        """Test step"""
        perf = self.step(batch, batch_idx)
        wd_distance = perf['wd']
        avg_nll = perf['nll']
        
        ## update and log metrics
        self.test_wd(wd_distance)
        self.test_nll(avg_nll)
        self.log("test/wd", wd_distance, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/nll", avg_nll, on_step=False, on_epoch=True, prog_bar=True)

        return perf, batch_idx
    
    def test_epoch_end(self, outputs: List[Any]):
        wd = self.test_wd.compute()
        self.test_wd_best(wd)
        self.log("test/wd_best", self.test_wd_best.compute(), prog_bar=True)
        
        ## similiarly for NLL
        nll = self.test_nll.compute()
        self.test_nll_best(nll)
        self.log("test/nll_best", self.test_nll_best.compute(), prog_bar=True)
        
        ## comparison
        if self.test_nll.compute() < self.test_nll_best.compute() or \
            self.test_wd.compute() < self.test_wd_best.compute():
            perf, batch_idx = outputs[0]
            outname = f"test-{self.current_epoch}-{batch_idx}"
            self.compare(perf, outname)
    