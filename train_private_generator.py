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

    explainer = VGG16_w_XAI_n_DP(num_classes=n_class, epsilon=args.eps, delta=args.delta)

    if os.path.exists(args.explainer):
        print(f'weight file - {args.explainer} is found.')

        explainer.load_state_dict(torch.load(args.explainer))

        print(f'explainer model loaded succesfully...')
    else:
        print(f'can not find weight file - {args.explainer}')
        exit(-1)

    explainer = explainer.eval()

    generator = Generator()
    discriminator = Discriminator()
    noiser = NN.GaussianNoiseAdder(args.std)

    if args.resume:
        print(f'resume training...')

        if os.path.exists(os.path.join(args.output_dir, args.generator)):
            print(f'previous weights are found.')

            generator.load_state_dict(torch.load(os.path.join(args.output_dir, args.generator)))

            print(f'previous weights are loaded succesfully.')

        else:
            print(f'previous weights are not found.')
            print(f'create new model...')

    explainer = explainer.to(device)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    noiser = noiser.to(device)

    if args.world_size > 0:
        generator = DDP(generator, device_ids=[rank])
        discriminator = DDP(discriminator, device_ids=[rank])

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda epoch: 0.95 ** epoch)
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda epoch: 0.95 ** epoch)

    if torch.cuda.is_available():
        scaler_G = amp.GradScaler()
        scaler_D = amp.GradScaler()

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
            target_real = torch.Tensor(target.size(0)).fill_(1.0)
            target_fake = torch.Tensor(target.size(0)).fill_(0.0)

            target_real = target_real.to(device)
            target_fake = target_fake.to(device)
            data = data.to(device)
            
            optimizer_G.zero_grad()

            if torch.cuda.is_available():
                with amp.autocast():
                    noised_data = noiser(data)
                    private_data = generator(data)

                    pred_fake = discriminator(private_data)
                    loss_GAN = F.mse_loss(pred_fake, target_real)

                    org_map, _, _ = explainer.explain(data, eXplainMethod.guided_backprop)
                    private_map, _, _ = explainer.explain(private_data, eXplainMethod.guided_backprop)

                    loss_map = F.l1_loss(private_data, org_map.detach())

                    loss_G = loss_GAN + loss_map

                scaler_G.scale(loss_G).backward()
                scaler_G.step(optimizer_G)
                scaler_G.update()
            else:
                noised_data = noiser(data)
                private_data = generator(data)

                pred_fake = discriminator(private_data)
                loss_GAN = F.mse_loss(pred_fake, target_real)

                org_map, _, _ = explainer.explain(data, eXplainMethod.guided_backprop)
                private_map, _, _ = explainer.explain(private_data, eXplainMethod.guided_backprop)

                loss_map = F.l1_loss(private_data, org_map.detach())

                loss_G = loss_GAN + loss_map

                loss_G.backward()
                optimizer_G.step()

            optimizer_D.zero_grad()

            if torch.cuda.is_available():
                with amp.autocast():
                    pred_real = discriminator(noised_data.detach())
                    loss_D_real = F.mse_loss(pred_real, target_real)

                    pred_fake = discriminator(private_data.detach())
                    loss_D_fake = F.mse_loss(pred_fake, target_fake)

                    loss_D = loss_D_real + loss_D_fake

                scaler_D.scale(loss_D).backward()
                scaler_D.step(optimizer_D)
                scaler_D.update()
            else:
                pred_real = discriminator(noised_data.detach())
                loss_D_real = F.mse_loss(pred_real, target_real)

                pred_fake = discriminator(private_data.detach())
                loss_D_fake = F.mse_loss(pred_fake, target_fake)

                loss_D = loss_D_real + loss_D_fake

                loss_D.backward()
                optimizer_D.step()

            if (args.world_size > 0 and rnak == 0) or (args.world_size <= 0):
                print(f'\rEpoch[{epoch+1}/{args.epochs}], Step[{i}/{total_step}], Loss_G[{loss_G.item():.5f}], loss_map[{loss_map.item():.5f}], loss_D[{loss_D.item():.5f}], runtime: {time.time() - start:.3f}(s)', end='')

    if args.world_size > 0 and rnak == 0:
        print(f'\ntrain completed in {time.time() - start:.3f}(s)')

        torch.save(generator.module.state_dict(), os.path.join(args.output_dir, args.generator))
    else:
        print(f'\ntrain completed in {time.time() - start:.3f}(s)')

        torch.save(generator.state_dict(), os.path.join(args.output_dir, args.generator))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='cifar100', help='')
    parser.add_argument('--input_size', type=int, default=224, help='')

    parser.add_argument('--explainer', type=str, default='./model/vgg16_cifar100_x224.pth', help='')

    parser.add_argument('--output_dir', type=str, default='./model', help='')
    parser.add_argument('--generator', type=str, default='vgg16_cifar100_x224_std_5.0_eps_0.1_delta_1e-5_generator.pth', help='')

    parser.add_argument('--epochs', type=int, default=30, help='')
    parser.add_argument('--batch_size', type=int, default=24, help='')
    parser.add_argument('--lr', type=float, default=2e-4, help='')

    parser.add_argument('--eps', type=float, default=0.1, help='')
    parser.add_argument('--delta', type=float, default=1e-5, help='')
    parser.add_argument('--std', type=float, default=5.0, help='')

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