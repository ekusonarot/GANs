import argparse
from model import GAN
import torch
from torchvision import datasets, transforms, utils
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

if __name__ == "__main__":
    gan = GAN()
    parser = argparse.ArgumentParser(prog="Generative Adversarial Network")
    parser.add_argument("model_name", type=str, choices=gan.names(), default="normal")
    parser.add_argument("-i", "--discriminator_iter", type=int, default=1)
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-6)
    parser.add_argument("-e", "--epoch", type=int, default=200)
    parser.add_argument("-b", "--batch", type=int, default=20)
    parser.add_argument("-g", "--gradient_penalty", action="store_true")
    
    args = parser.parse_args()
    gan.load(args.model_name, args.learning_rate, args.discriminator_iter, args.gradient_penalty)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gan = gan.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(28)
    ])
    train_dataset = datasets.CIFAR10(
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
            loss = gan.one_epoch(x)
            if j % 10 == 0:
                img = gan(64)
                grid = utils.make_grid(img)
                save_image(grid, f"checkpoint/epoch{i}_step{j}.png")
                torch.save(gan.state_dict(), f"checkpoint/epoch{i}_step{j}.pth")
                writer.add_image("images", grid, j)
            writer.add_scalars(f"loss/{i}", loss, j)
        