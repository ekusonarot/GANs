import argparse
from model import GAN
import torch
from torchvision import datasets, transforms, utils
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gan = GAN(device)
    parser = argparse.ArgumentParser(prog="Generative Adversarial Network")
    parser.add_argument("model_name", type=str, choices=gan.names())
    parser.add_argument("-i", "--discriminator_iter", type=int, default=5)
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("-e", "--epoch", type=int, default=200)
    parser.add_argument("-b", "--batch", type=int, default=20)
    
    args = parser.parse_args()
    gan.load(args.model_name, args.learning_rate, args.discriminator_iter)

    gan = gan.to(device)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = datasets.MNIST(
        './data',
        train = True,
        download = True,
        transform = transform
        )
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = args.batch,
        shuffle = True)
    
    writer = SummaryWriter()
    for i in range(args.epoch):
        for j, (x, _) in enumerate(tqdm(dataloader)):
            x = x.to(device)
            gan.one_epoch(x)
            if j % 100 == 0:
                noise = torch.randn(size=(64, 100, 1, 1)).to(device)
                img = gan(noise)
                grid = utils.make_grid(img)
                writer.add_image("images", grid, j)
        