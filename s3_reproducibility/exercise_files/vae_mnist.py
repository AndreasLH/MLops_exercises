"""
Adapted from
https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb

A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST
"""
import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Encoder, Decoder, Model
import hydra

from omegaconf import OmegaConf

import logging
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(cfg):
    log.info(f"configuration: \n {OmegaConf.to_yaml(cfg)}")
    print("Working directory : {}".format(os.getcwd()))
    hparams = cfg.experiment.hparams # not 100% correct

    # Model Hyperparameters
    dataset_path = hparams.dataset_path
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = hparams.batch_size
    x_dim  = hparams.x_dim
    hidden_dim = hparams.hidden_dim
    latent_dim = hparams.latent_dim
    lr = hparams.lr
    epochs = hparams.epochs
    torch.manual_seed(hparams.seed)
    

    # Data loading
    mnist_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
    test_dataset  = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)

    encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

    model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)

    from torch.optim import Adam

    BCE_loss = nn.BCELoss()

    def loss_function(x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

        return reproduction_loss + KLD

    optimizer = Adam(model.parameters(), lr=lr)


    log.info("Start training VAE...")
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        log.info(f"Epoch {epoch + 1} complete! Average Loss: { overall_loss / (batch_idx*batch_size)}")    
    log.info("Finish!!")

    # save weights
    torch.save(model, f"{os.getcwd()}/trained_model.pt")

    # Generate reconstructions
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)      
            x_hat, _, _ = model(x)       
            break

    save_image(x.view(batch_size, 1, 28, 28), 'orig_data.png')
    save_image(x_hat.view(batch_size, 1, 28, 28), 'reconstructions.png')

    # Generate samples
    with torch.no_grad():
        noise = torch.randn(batch_size, latent_dim).to(DEVICE)
        generated_images = decoder(noise)
        
    save_image(generated_images.view(batch_size, 1, 28, 28), 'generated_sample.png')


if __name__ == "__main__":
    main()