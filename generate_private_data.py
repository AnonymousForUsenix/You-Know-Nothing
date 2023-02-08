import os
import time
import argparse

import numpy as np

import torch

import torchvision.transforms as transforms

from PIL import Image

from modules.network import *
from datasets import *

def generate(args):
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
        dataset = CIFAR10(transform=transform, mode='test')
    elif args.dataset == 'cifar100':
        dataset = CIFAR100(transform=transform, mode='test')
    else:
        print(f'dataset - {args.dataset} is not defined.')
        exit(-1)

    generator = Generator()

    if os.path.exists(args.generator):
        print(f'weight file - {args.generator} is found.')

        generator.load_state_dict(torch.load(args.generator))

        print(f'generator model loaded succesfully...')
    else:
        print(f'can not find weight file - {args.generator}')
        exit(-1)

    generator.to(device)

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.generator))[0])):
        os.makedirs(os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.generator))[0]))

    idx = 0
    start = time.time()
    total_batch = len(dataloader)
    for i, (data, target) in enumerate(dataloader):
        data = data.to(device)

        with torch.no_grad():
            private_data = generator(data)

        for j in range(private_data.size(0)):
            img = private_data[j, :, :, :].clone()
            img = 0.5 * (img + 1.0)
            img = np.array(transforms.ToPILImage()(img.data.squeeze(0).cpu()))

            Image.fromarray(img).save(os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.generator))[0]) + '/' + str(idx) + '_' + format(target[j].item(), 'd') + args.ext)

            idx += 1

        print(f'\rBatch[{i}/{total_batch}], runtime: {time.time() - start:.3f}(s)', end='')

    print(f'\nprivate image generated in {time.time() - start:.3f}(s)')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='cifar100', help='')
    parser.add_argument('--input_size', type=int, default=224, help='')

    parser.add_argument('--generator', type=str, default='./model/vgg16_cifar100_x224_std_5.0_eps_0.1_delta_1e-5_generator.pth', help='')

    parser.add_argument('--output_dir', type=str, default='./output', help='')

    parser.add_argument('--batch_size', type=int, default=128, help='')

    parser.add_argument('--ext', type=str, default='.png', help='file extention')

    args, _ = parser.parse_known_args()

    if _:
        print('Unparsed arguments', _)
        exit(-1)

    generate(args)

if __name__ == '__main__':
    main()
