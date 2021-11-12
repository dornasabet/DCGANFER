import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Generator, Discriminator, weight_initialize
import config
import torch.optim as optim
import torchvision
from dataset import Dataset_fer
import mlflow
from os.path import join
import os
from torchvision.utils import save_image
from utils import load_checkpoint, save_checkpoint

# mlflow.set_tracking_uri(config.MLFLOW_SOURCE)
# mlflow.set_experiment(config.MLFLOW_EXP)
# mlflow.start_run()

os.makedirs(config.SAVE_PATH, exist_ok=True)

# loss
criterion = nn.BCELoss()
# optimizer

gen = Generator(config.NOISE_DIM, config.IMG_CHANNEL, config.GEN_FEATURES).to(config.DEVICE)
disc = Discriminator(config.IMG_CHANNEL, config.DISC_FEATURES).to(config.DEVICE)

weight_initialize(gen)
weight_initialize(disc)

gen.train()
disc.train()

optimizerD = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
optimizerG = optim.Adam(gen.paramers(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

dataset = Dataset_fer(root=config.DATA_DIR, transform=config.TRANSFORM)
dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS,
                        pin_memory=True)

fixed_noise = torch.randn(32, config.NOISE_DIM, 1, 1).to(config.DEVICE)

step = 0

writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
writer_loss = SummaryWriter(f"logs")

if config.START_EPOCH:
    load_checkpoint(config.GEN_CHECKPOINT, gen, optimizerG, config.LEARNING_RATE)
    load_checkpoint(config.DISC_CHECKPOINT, disc, optimizerD, config.LEARNING_RATE)

for epoch in range(config.NUM_EPOCH):
    for idx, real in enumerate(dataloader):
        real = real.to(config.DEVICE)
        noise = torch.randn(config.BATCH_SIZE, config.NOISE_DIM, 1, 1).to(config.DEVICE)
        fake = gen(noise)

        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))

        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        loss_disc = (loss_disc_fake + loss_disc_real) / 2

        # mlflow.log_metric("loss_disc_real", loss_disc_real.item())
        # mlflow.log_metric("loss_disc_fake", loss_disc_fake.item())
        # mlflow.log_metric("D_loss", loss_disc.item())

        disc.zero_grad()
        loss_disc.backward()
        optimizerD.step()

        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))

        # mlflow.log_metric("loss_gen", loss_gen.item())

        gen.zero_grad()
        loss_gen.backward()
        optimizerG.step()

        if idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{config.NUM_EPOCH}] Batch {idx}/{len(dataloader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )
            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                writer_real.add_image("Real", img_grid_real, step=step)
                writer_fake.add_image("Fake", img_grid_fake, step=step)

                path = f"{config.SAVE_PATH}/{epoch}_{idx}.png"
                save_image(fake, path)
                # mlflow.log_artifact(path, join('images', path))

            step += 1

    writer_loss.add_scalar("loss_disc_real", loss_disc_real, epoch)
    writer_loss.add_scalar("loss_disc_fake", loss_disc_fake, epoch)
    writer_loss.add_scalar("loss_disc", loss_disc, epoch)
    writer_loss.add_scalar("loss_gen", loss_gen, epoch)
    save_checkpoint(gen, optimizerG, file_name=config.GEN_CHECKPOINT)
    save_checkpoint(disc, optimizerD, file_name=config.DISC_CHECKPOINT)
