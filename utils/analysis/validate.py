import sys
import torch
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from sklearn.manifold import TSNE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate(val_loader, model, domain_discriminator, visualize = False):
    '''Returns the mean validation loss (regressor) and accuracy (domain discriminator) for the validation set given the model.
    If visualize = True, the features of the validation set are collected and visualized using t-SNE.'''
    # send model to device
    model.to(device)
    domain_discriminator.to(device)

    # switch to evaluate mode
    model.eval()
    domain_discriminator.eval()

    # initialize loss and accuracy
    val_loss = 0
    val_acc = 0

    for i in range(len(val_loader)):
        # fetch data
        images, scores = next(iter(val_loader))

        # send data to device
        images = images.to(device)
        scores = scores.to(device) # the scores for the source images are between 0 and 1 and for target NaNs (unknown)
        labels = scores.clone()
        labels = [1 if x > 0 else 0 for x in labels].to(device) # create list with 1 for source and 0 for target

        # compute output
        y, f = model(images)

        print(y.shape)
        print(f.shape)

        # compute validation loss
        val_loss += F.mse_loss(scores, y)

        # compute domain labels
        d_labels = domain_discriminator(images)

        # compute accuracy
        val_acc += torch.sum(d_labels == labels).item() / len(labels)

        if visualize:
            # initialize tsne
            tsne = TSNE(n_components=2)
            # collect features
            tsne_result = tsne.fit_transform(f.detach().cpu().numpy())
            tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': labels.detach().cpu().numpy()})
            fig, ax = plt.subplots(1)
            sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120)
            lim = (tsne_result.min()-5, tsne_result.max()+5)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect('equal')
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

            # save figure
            fig.savefig('results/tsne.png', bbox_inches='tight')

    return val_loss, val_acc

            



    return top1.avg