import os
import time
import argparse

import torch
import torch.nn.functional as F

import torchvision.transforms as transforms

from modules.network import *
from datasets import *

def test(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')

        print(f'use cuda as backend...')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')

        print(f'use mps as backend...')
    else:
        device = torch.device('cpu')

        print(f'use cpu as backend...')

    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if args.dataset == 'cifar10':
        n_class = 10
        dataset = CIFAR10(transform=transform, mode='test')
    elif args.dataset == 'cifar100':
        n_class = 100
        dataset = CIFAR100(transform=transform, mode='test')
    else:
        print(f'dataset - {args.dataset} is not defined.')
        exit(-1)

    model = VGG16_w_XAI(num_classes=n_class)

    if os.path.exists(args.model):
        print(f'weight file - {args.model} is found.')

        model.load_state_dict(torch.load(args.model))

        print(f'model loaded succesfully...')
    else:
        print(f'can not find weight file - {args.model}')
        exit(-1)

    generator = Generator()

    if os.path.exists(args.generator):
        print(f'weight file - {args.generator} is found.')

        generator.load_state_dict(torch.load(args.generator))

        print(f'generator model loaded succesfully...')
    else:
        print(f'can not find weight file - {args.generator}')
        exit(-1)

    model = model.eval()
    generator = generator.eval()

    model = model.to(device)
    generator = generator.to(device)

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size)

    n_test = 0
    n_correct = 0

    start = time.time()
    total_step = len(dataloader)
    for i, (data, target) in enumerate(dataloader):
        data = data.to(device)
        target = target.to(device)

        with torch.no_grad():
            private_data = generator(data)
            y = model(private_data)

        _, prediction = torch.max(y.data, 1)

        n_test += target.size(0)
        n_correct += (prediction == target).sum().item()

        print(f'\rStep[{i+1}/{total_step}], runtime {time.time() - start:.3f}(s)', end='')

    print(f'\ntest completed in {time.time() - start:.3f}(s)')
    print(f'\ntest accuracy is {n_correct/n_test:.5f}')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='cifar100', help='')
    parser.add_argument('--input_size', type=int, default=224, help='')

    parser.add_argument('--model', type=str, default='./model/vgg16_cifar100_x224_std_5.0_eps_0.1_delta_1e-5.pth', help='')
    parser.add_argument('--generator', type=str, default='./model/vgg16_cifar100_x224_std_5.0_eps_0.1_delta_1e-5_generator.pth', help='')

    parser.add_argument('--batch_size', type=int, default=96, help='')

    args, _ = parser.parse_known_args()

    if _:
        print('Unparsed arguments', _)
        exit(-1)

    test(args)

if __name__ == '__main__':
    main()