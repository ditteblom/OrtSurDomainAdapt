import argparse
import os.path as osp
import time

import torchvision.transforms as transforms
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from utils.misc.data_loader import get_loader
from utils.misc.logger import CompleteLogger
from utils.modules.regressor import Regressor
from utils.modules.domain_discriminator import DomainDiscriminator
from utils.misc.data_iter import ForeverDataIterator
from utils.misc.da_loss import DomainAdversarialLoss
from utils.meter import AverageMeter, ProgressMeter
import utils.models as models
from utils.analysis import collect_feature, tsne, a_distance, validate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    logger = CompleteLogger(args.log, args.phase)
    size = 224

    # data augmentation
    train_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((size,size)),
                                    #transforms.RandomHorizontalFlip(),
                                    transforms.RandomChoice([
                                    #transforms.GaussianBlur(kernel_size=3, sigma=(1, 1)),
                                    transforms.RandomRotation(degrees=(15)),])
                                    ])
    val_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((size,size)),
                                    ])

    # Data loaders.
    train_source_loader = get_loader(args.repair_type, 'train', 'source', args.data_dir, args.batch_size, transform = train_transform)
    train_target_loader = get_loader(args.repair_type, 'train', 'target', args.data_dir, args.batch_size, transform = train_transform) #add target (real Xrays) data loader

    val_loader = get_loader(args.repair_type, 'val', 'source', args.data_dir, args.batch_size, transform = val_transform)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    print('Data loaded. Start training...')

    # create model
    #print("=> using pre-trained model '{}'".format(args.arch))
    backbone = models.resnet18(pretrained=True)
    backbone.fc = nn.Identity()

    # setup regressor and domain discriminator
    regressor = Regressor(backbone=backbone, num_factors=1).to(device)
    domain_discri = DomainDiscriminator(in_feature=regressor.features_dim, hidden_size=1024).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(regressor.get_parameters() + domain_discri.get_parameters(), args.lr, momentum=args.momentum,
                    weight_decay=args.wd, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    dann = DomainAdversarialLoss(domain_discri).to(device)

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location=device)
        regressor.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(regressor.backbone, regressor.pool_layer, regressor.bottleneck).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.png')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = utils.validate(test_loader, classifier, args, device)
        print(acc1)
        return

    # start training
    best_mae = 100000.
    for epoch in range(args.epochs):
        # train for one epoch
        print("lr", lr_scheduler.get_lr())
        train(train_source_iter, train_target_iter, regressor, dann, optimizer,
              lr_scheduler, epoch, args)

        # evaluate on validation set
        mae = validate(val_loader, regressor, args, 1, device)

        # remember best mae and save checkpoint
        torch.save(regressor.state_dict(), logger.get_checkpoint_path('latest'))
        if mae < best_mae:  
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_mae = min(mae, best_mae)
        print("mean MAE {:6.3f} best MAE {:6.3f}".format(mae, best_mae))

    print("best_mae = {:6.3f}".format(best_mae))

    logger.close()

def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          model: Regressor, domain_adv: DomainAdversarialLoss, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, domain_accs],
        prefix="Epoch: [{}]".format(epoch+1))

    # switch to train mode
    model.train()
    domain_adv.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_t = next(train_target_iter)

        # send to device
        x_s, labels_s = x_s.to(device), labels_s.to(device)
        x_t = x_t.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_s, _ = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        cls_loss = F.mse_loss(y_s.squeeze(), labels_s)
        transfer_loss = domain_adv(f_s, f_t)
        domain_acc = domain_adv.domain_discriminator_accuracy
        loss = cls_loss + transfer_loss * args.trade_off


        losses.update(loss.item(), x_s.size(0))
        domain_accs.update(domain_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DANN for Regression Domain Adaptation')
    # dataset parameters
    parser.add_argument('--resize-size', type=int, default=128)
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=12, type=int,
                        metavar='N',
                        help='mini-batch size (default: 36)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.001, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=300, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=44, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--log", type=str, default='dann',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")

    # Model configuration.
    parser.add_argument('--repair_type', type=str, default='028_cannulated_screws', help='type of fracture repair')
    parser.add_argument('--actfun', type=str, default='relu', help='type of activation function')
    parser.add_argument('--alpha', type=float, default='1', help='alpha for elu activation function')
    
    # Training configuration.
    parser.add_argument('--data_dir', type=str, default='Data/')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--num_iters', type=int, default=1000, help='number of total iterations')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate for optimizer')
    
    # Miscellaneous.
    parser.add_argument('--log_step', type=int, default=1)
    parser.add_argument('--run_name', type=str, default=datetime.now().strftime('%y%B%d_%H%M_%S'))

    args = parser.parse_args()
    print(args)
    main(args)