
import os
import pickle
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import torch
from pytorch_lightning import LightningDataModule

class Herwig(LightningDataModule):
    def __init__(
        self, 
        data_dir: str = "data/",
        fname: str = "allHadrons_10M_mode4_with_quark_with_pert.npz",
        original_fname: str = "cluster_ML_allHadrons_10M.txt",
        train_val_test_split: Tuple[int, int, int] = (100, 50, 50),
        num_output_hadrons: int = 2,
        num_particle_kinematics: int = 2,
        # hadron_type_embedding_dim: int = 10,
    ):
        """This is for the GAN datamodule"""
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.cond_dim: Optional[int] = None
        self.output_dim: Optional[int] = None
        
        self.pids_to_ix: Optional[Dict[int, int]] = None
        
        ## particle type map
        self.pids_map_fname = os.path.join(self.hparams.data_dir, "pids_to_ix.pkl")
        self.num_hadron_types: int = 0
    
    
    def prepare_data(self):
        ## read the original file, determine the number of particle types
        ## and create a map.
        if os.path.exists(self.pids_map_fname):
            print("Loading existing pids map")
            self.pids_to_ix = pickle.load(open(self.pids_map_fname, 'rb'))
            self.num_hadron_types = len(list(self.pids_to_ix.keys()))
            print("END...Loading existing pids map")
        else:
            fname = os.path.join(self.hparams.data_dir, self.hparams.origin_fname)
            if not os.path.exists(fname):
                raise FileNotFoundError(f"File {fname} not found.")
            df = pd.read_csv(fname, sep=';', header=None, names=None, engine='python')
            
            def split_to_float(df, sep=','):
                out = df
                if type(df.iloc[0]) == str:
                    out = df.str.split(sep, expand=True).astype(np.float32)
                return out
            
            q1,q2,c,h1,h2 = [split_to_float(df[idx]) for idx in range(5)]
            h1_type, h2_type = h1[[0]], h2[[0]]
            hadron_pids = np.unique(np.concatenate([h1_type, h2_type])).astype(np.int64)
            
            self.pids_to_ix = {pids: i for i, pids in enumerate(hadron_pids)}
            self.num_hadron_types = len(hadron_pids)
            
            pickle.dump(self.pids_to_ix, open(self.pids_map_fname, "wb"))
            
            
    def create_dataset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """"It creates the dataset for training a conditional GAN.
        Returns:
            cond_info: conditional information
            x_truth:   target truth information with conditonal information
        """
        fname = os.path.join(self.hparams.data_dir, self.hparams.fname)
        arrays = np.load(fname)
        
        cond_info = torch.from_numpy(arrays['cond_info'].astype(np.float32))
        truth_in = torch.from_numpy(arrays['out_truth'].astype(np.float32))
        
        num_tot_evts, self.cond_dim = cond_info.shape
        num_asked_evts = sum(self.hparams.train_val_test_split)
        
        print(f"Number of events: {num_tot_evts:,}, asking for {num_asked_evts:,}")
        if num_tot_evts < num_asked_evts:
            raise ValueError(f"Number of events {num_tot_evts} is less than asked {num_asked_evts}")
        
        cond_info = cond_info[:num_asked_evts]
        truth_in = truth_in[:num_asked_evts]
        
        
        ## output includes N hadron types and their momenta
        ## output dimension only includes the momenta
        self.output_dim = truth_in.shape[1] - self.hparams.num_output_hadrons

        true_hadron_momenta = truth_in[:, :-self.hparams.num_output_hadrons]
           
        ## convert particle IDs to indices
        ## then these indices can be embedded in N dim. space
        target_hadron_types = truth_in[:, -self.hparams.num_output_hadrons:].reshape(-1).long()
        target_hadron_types_idx = torch.from_numpy(np.vectorize(
            self.pids_to_ix.get)(target_hadron_types.numpy())).reshape(-1, self.hparams.num_output_hadrons)
        
        self.summarize()
        return (cond_info, true_hadron_momenta, target_hadron_types_idx)
    
    
    def summarize(self):
        print(f"Reading data from: {self.hparams.data_dir}")
        print(f"\tNumber of hadron types: {self.num_hadron_types}")
        print(f"\tNumber of conditional variables: {self.cond_dim}")
        print(f"\tNumber of output variables: {self.output_dim}")
        print(f"\tNumber of output hadrons: {self.hparams.num_output_hadrons}")
        print(f"\tNumber of particle kinematics: {self.hparams.num_particle_kinematics}")
        