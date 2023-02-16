import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(train_source_iter, train_target_iter,
          model, domain_adv, optimizer,
          lr_scheduler, epoch: int, args):

    # switch to train mode
    model.train()
    domain_adv.train()

    for i in range(args.iters_per_epoch):
        # get data
        x_s, labels_s = next(train_source_iter)
        x_t = next(train_target_iter)

        # send to device
        x_s, labels_s = x_s.to(device), labels_s.to(device)
        x_t = x_t.to(device)

        # compute output
        x = torch.cat((x_s, x_t), dim=0) # concatenate source and target data into one big batch
        y, f = model(x) # pass through model, y is the score, f are the features
        y_s, y_t = y.chunk(2, dim=0) # split the scores into source and target
        f_s, f_t = f.chunk(2, dim=0) # split the features into source and target

        # compute loss
        reg_loss = F.mse_loss(y_s.squeeze(), labels_s) # compute regression loss (only on source data)
        transfer_loss = domain_adv(f_s, f_t) # compute domain adversarial loss
        domain_acc = domain_adv.domain_discriminator_accuracy
        loss = reg_loss + transfer_loss * args.trade_off

        # compute gradients and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if i % args.print_freq == 0:
            print(f'Epoch: [{i}/{args.iters_per_epoch}], Regression loss: {reg_loss.item()}, Transfer loss: {transfer_loss.item()}, Total loss: {loss}, Domain acc.: {domain_acc.item()}\n')
