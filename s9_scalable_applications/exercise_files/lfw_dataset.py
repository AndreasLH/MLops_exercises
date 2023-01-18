"""
LFW dataloading
"""
import argparse
import glob
import time
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.io import read_image

plt.rcParams["savefig.bbox"] = 'tight'


class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        # TODO: fill out with what you need
        self.imgs = glob.glob(path_to_folder+"/*/*.jpg")
        self.transform = transform
        
    def __len__(self):
        return len(self.imgs) # TODO: fill out
    
    def __getitem__(self, index: int) -> torch.Tensor:
        # TODO: fill out
        img = Image.open(self.imgs[index])
        return self.transform(img)


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

def timing(make_experiment=False, num_workers=0):
    if make_experiment:
        dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=num_workers
        )
    res = [ ]
    for _ in range(5):
        start = time.perf_counter()
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx > args.batches_to_check:
                break
        end = time.perf_counter()

        res.append(end - start)
        
    res = np.array(res)
    print(f'Timing: {np.mean(res)}+-{np.std(res)}')
    return np.mean(res), np.std(res)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='lfw-deepfunneled', type=str)
    parser.add_argument('-batch_size', default=128, type=int)
    parser.add_argument('-num_workers', default=0, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-make_error_bar_plot', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    parser.add_argument('-batches_to_check', default=100, type=int)
    
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    if args.visualize_batch:
        imgs = next(iter(dataloader))    
        grid = make_grid(imgs)
        show(grid)
        
    if args.get_timing:
        # lets do some repetitions
        timing()

    if args.make_error_bar_plot:
        y, yerr = [], []
        x = range(4)
        for num_workers in tqdm(x):
            mean, std = timing(make_experiment=True, num_workers=num_workers)
            y.append(mean)
            yerr.append(std)
        plt.errorbar(x, y, yerr)
        plt.title(f"Time for loading data, batch size {args.batch_size}")
        plt.xlabel("n_workers")
        plt.ylabel("Time (s)")
        plt.savefig("dataloading_experiment.png",dpi=200)
