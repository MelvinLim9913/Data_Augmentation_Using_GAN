import argparse
import os
import numpy as np
import torch
import torch.autograd as autograd
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision.utils import save_image

from wgan_gp_component import Generator, Discriminator, MyDataset

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=2000,
                    help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128,
                    help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0015,
                    help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100,
                    help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=255,
                    help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3,
                    help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5,
                    help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01,
                    help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int,
                    default=400, help="interval betwen image samples")
parser.add_argument("--dataset", type=str, default="original",
                    choices=["original", "esrgan"],
                    help="which dataset to train on")
opt = parser.parse_args()
print(opt)

dataroot = "dataset/original/train" if opt.dataset == "original" else "dataset/esrgan/train"
output_dir = "dataset/wgan_gp" if opt.dataset == "original" else "dataset/esrgan_wgan_gp"
weights_dir = "weights/wgan_gp" if opt.dataset == "original" else "weights/esrgan_wgan_gp"

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

# Loss weight for gradient penalty
lambda_gp = 10


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha)
                                            * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(
        1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


for class_number in range(7):
    sample_images_dir = os.path.join(
        weights_dir, str(class_number), "sample_images")
    class_weights_dir = os.path.join(weights_dir, str(class_number))

    os.makedirs(sample_images_dir, exist_ok=True)
    os.makedirs(class_weights_dir, exist_ok=True)

    # Configure data loader
    dataset = MyDataset(path=os.path.join(dataroot, str(class_number)),
                        transform=transforms.Compose([
                            transforms.Resize(opt.img_size),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])
                        ])
                        )
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                             shuffle=True, drop_last=True)

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    batches_done = 0
    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(
                0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            fake_imgs = generator(z)

            # Real images
            real_validity = discriminator(real_imgs)
            # Fake images
            fake_validity = discriminator(fake_imgs)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(
                discriminator, real_imgs.data, fake_imgs.data)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + \
                     torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_imgs = generator(z)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

                if batches_done % opt.sample_interval == 0:
                    save_image(
                        fake_imgs.data[:25],
                        os.path.join(sample_images_dir,
                                     f"{class_number}/{batches_done}.png"),
                        nrow=5,
                        normalize=True
                    )
                    torch.save(generator.state_dict(), os.path.join(
                        class_weights_dir, f"{class_number}/{batches_done}.pth"))
                    print(f"Saved images and weight, {batches_done}")

                batches_done += opt.n_critic
