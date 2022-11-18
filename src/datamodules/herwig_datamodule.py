import os
import pickle
from typing import Any, Dict, Optional, Tuple

## for data processing
import numpy as np
import pandas as pd



import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, TensorDataset

class HerwigAllHadronsDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/herwig",
        fname: str = "allHadrons_10M_mode4_with_quark_with_pert.npz",
        origin_fname: str = "cluster_ML_allHadrons_10M.txt",
        train_val_test_split: Tuple[int, int, int] = (1_000_000, 50_000, 50_000),
        batch_size: int = 5000,
        num_workers: int = 12,
        pin_memory: bool = False,
    ):
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
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
            
    
    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        if not self.data_train and not self.data_val and not self.data_test:
            fname = os.path.join(self.hparams.data_dir, self.hparams.fname)
            arrays = np.load(fname)
            truth_in = torch.from_numpy(arrays['out_truth'].astype(np.float32))
            cond_info = torch.from_numpy(arrays['cond_info'].astype(np.float32))
            dataset = TensorDataset(truth_in, cond_info)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )
            
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
        
    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
    
    
if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "herwig.yaml")
    cfg.data_dir = str(root / "data" / "herwig")
    _ = hydra.utils.instantiate(cfg)