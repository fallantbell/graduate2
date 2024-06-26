from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from data_loader.dataset import ACIDdataset
from data_loader.re10k_dataset import Re10k_dataset


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class ACIDDataLoader(DataLoader):
    """
    ACID dataloader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, mode="train"):

        self.data_dir = data_dir

        if mode == "train":
            self.dataset = ACIDdataset(self.data_dir,"train")
        elif mode == "validation":
            self.dataset = ACIDdataset(self.data_dir,"validation")
        else:
            self.dataset = ACIDdataset(self.data_dir,"test")
        

        collate_fn=default_collate

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }

        super().__init__(**self.init_kwargs)

class re10k_DataLoader(DataLoader):
    """
    ACID dataloader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, 
                validation_split=0.0, num_workers=1, 
                mode="train",max_interval = 5,midas_transform = None,
                infer_len = 20, do_latent = True,
                ):

        self.data_dir = data_dir

        if mode == "train":
            self.dataset = Re10k_dataset(self.data_dir,"train",max_interval,midas_transform,do_latent = do_latent)
        elif mode == "validation":
            self.dataset = Re10k_dataset(self.data_dir,"validation")
        else:
            self.dataset = Re10k_dataset(self.data_dir,"test",midas_transform = midas_transform,infer_len=infer_len,do_latent = do_latent)
        

        collate_fn=default_collate

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            "drop_last": True,
        }

        super().__init__(**self.init_kwargs)