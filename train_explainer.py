import os
import time
import argparse

import torch
import torch.nn.functional as F

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

if torch.cuda.is_available():
    import torch.cuda.amp as amp

import torchvision.transforms as transforms

from modules.network import *
from datasets import *

def train(rank, args):
    if args.world_size > 0:
        dist.init_process_group('nccl', rank=rank, world_size=args.world_size)
        device = torch.device('cuda:' + str(rank))

        print(f'use distributed data parallel...')
    elif torch.cuda.is_available():
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
        transforms.RandomCrop(args.input_size, padding=int(args.input_size/8)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if args.dataset == 'cifar10':
        n_class = 10
        dataset = CIFAR10(transform=transform, mode='train')
    elif args.dataset == 'cifar100':
        n_class = 100
        dataset = CIFAR100(transform=transform, mode='train')
    else:
        print(f'dataset - {args.dataset} is not defined.')
        exit(-1)

    model = VGG16_w_XAI(num_classes=n_class, input_size=args.input_size)

    if args.resume:
        print(f'resume training...')

        if os.path.exists(os.path.join(args.output_dir, args.weights_file)):
            print(f'previous weights are found.')

            model.load_state_dict(torch.load(os.path.join(args.output_dir, args.weights_file)))

            print(f'previous weights are loaded succesfully.')

        else:
            print(f'previous weights are not found.')
            print(f'create new model...')

    model.to(device)

    if args.world_size > 0:
        model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if torch.cuda.is_available():
        scaler = amp.GradScaler()

    if args.world_size > 0:
        data_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, sampler = data_sampler)
    else:
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size)

    start = time.time()
    total_step = len(dataloader)
    for epoch in range(args.epochs):
        if args.world_size > 0:
            data_sampler.set_epoch(epoch)

        for i, (data, target) in enumerate(dataloader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            if torch.cuda.is_available():
                with amp.autocast():
                    y = model(data)
                    loss = F.cross_entropy(y, target)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                y = model(data)
                loss = F.cross_entropy(y, target)

                loss.backward()
                optimizer.step()

            if (args.world_size > 0 and rnak == 0) or (args.world_size <= 0):
                print(f'\rEpoch[{epoch+1}/{args.epochs}], Step[{i}/{total_step}], Loss {loss.item():.5f}, runtime {time.time() - start:.3f}(s)', end='')

        lr_scheduler.step()

        if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

    if args.world_size > 0 and rnak == 0:
        print(f'\ntrain completed in {time.time() - start:.3f}(s)')

        torch.save(model.module.state_dict(), os.path.join(args.output_dir, args.weights_file))
    else:
        print(f'\ntrain completed in {time.time() - start:.3f}(s)')

        torch.save(model.state_dict(), os.path.join(args.output_dir, args.weights_file))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='cifar100', help='')
    parser.add_argument('--input_size', type=int, default=224, help='')

    parser.add_argument('--output_dir', type=str, default='./model', help='')
    parser.add_argument('--weights_file', type=str, default='vgg16_cifar100_x224.pth', help='')

    parser.add_argument('--epochs', type=int, default=50, help='')
    parser.add_argument('--batch_size', type=int, default=96, help='')
    parser.add_argument('--lr', type=float, default=1e-4, help='')
    
    parser.add_argument('--world_size', type=int, default=0, help='')

    parser.add_argument('--resume', action='store_true')

    args, _ = parser.parse_known_args()

    if _:
        print('Unparsed arguments', _)
        exit(-1)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    if args.world_size > 0:
        mp.spawn(train, args=(args,), nprocs=args.world_size, join=True)
    else:
        train(-1, args)

if __name__ == '__main__':
    main()
