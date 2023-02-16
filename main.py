import argparse
import os.path as osp
import time
import shutil

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
from utils.analysis.validate import validate
from utils.analysis.collect_feature import collect_feature
from utils.analysis.tsne import visualize
from utils.analysis.a_distance import calculate
from utils.training.train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    logger = CompleteLogger(args.log, args.phase)
    size = 224

    # data augmentation
    train_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((size,size)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomChoice([
                                    transforms.GaussianBlur(kernel_size=3, sigma=(1, 1)),
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
    val_iter = ForeverDataIterator(val_loader)

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
    

    if args.phase == 'test':
        (test_loss, test_acc) = validate(test_loader, regressor, args, device)
        print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}')

    if args.phase == 'train':
        # start training
        for epoch in range(args.epochs):
            # train for one epoch
            print("lr", lr_scheduler.get_lr())
            train(train_source_iter, train_target_iter, regressor, dann, optimizer,
                lr_scheduler, epoch, args)

            # evaluate on validation set
            (val_loss, val_acc) = validate(val_loader, regressor, dann)
            print(f'Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}')

            # remember best mae and save checkpoint
            #torch.save(regressor.state_dict(), logger.get_checkpoint_path('latest'))
            #if mae < best_mae:  
            #    shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
            #best_mae = min(mae, best_mae)
            #print("mean MAE {:6.3f} best MAE {:6.3f}".format(mae, best_mae))

        #print("best_mae = {:6.3f}".format(best_mae))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DANN for Regression Domain Adaptation')
    # dataset parameters
    parser.add_argument('--resize-size', type=int, default=224)
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
    parser.add_argument('-i', '--iters-per-epoch', default=100, type=int,
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
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--batch_size', type=int, default=6, help='batch size')
    parser.add_argument('--num_iters', type=int, default=1000, help='number of total iterations')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate for optimizer')
    
    # Miscellaneous.
    parser.add_argument('--run_name', type=str, default=datetime.now().strftime('%y%B%d_%H%M_%S'))

    args = parser.parse_args()
    print(args)
    main(args)