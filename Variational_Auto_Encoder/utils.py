################################################################################
# MIT License
#
# Copyright (c) 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Autumn 2022
# Date Created: 2022-11-25
################################################################################

import torch
from torchvision.utils import make_grid
import numpy as np


def sample_reparameterize(mean, std):
    """
    Perform the reparameterization trick to sample from a distribution with the given mean and std
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions
        std - Tensor of arbitrary shape with strictly positive values. Denotes the standard deviation
              of the distribution
    Outputs:
        z - A sample of the distributions, with gradient support for both mean and std.
            The tensor should have the same shape as the mean and std input tensors.
    """
    assert not (std < 0).any().item(), "The reparameterization trick got a negative std as input. " + \
                                       "Are you sure your input is std and not log_std?"
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    eps = torch.randn_like(mean)
    z = mean + std * eps
    #######################
    # END OF YOUR CODE    #
    #######################
    return z


def KLD(mean, log_std):
    """
    Calculates the Kullback-Leibler divergence of given distributions to unit Gaussians over the last dimension.
    See the definition of the regularization loss in Section 1.4 for the formula.
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions.
        log_std - Tensor of arbitrary shape and range, denoting the log standard deviation of the distributions.
    Outputs:
        KLD - Tensor with one less dimension than mean and log_std (summed over last dimension).
              The values represent the Kullback-Leibler divergence to unit Gaussians.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    sigma_sq = torch.exp(2*log_std)                  # sigma^2
    kld = 0.5 * (sigma_sq + mean**2 - 1 - 2*log_std) # 0.5 * (sigma^2 + mu^2 - 1 - 2*log(sigma))

    KLD = kld.sum(dim=-1)                            # sum over the last dimension
    #######################
    # END OF YOUR CODE    #
    #######################
    return KLD


def elbo_to_bpd(elbo, img_shape):
    """
    Converts the summed negative log likelihood given by the ELBO into the bits per dimension score.
    Inputs:
        elbo - Tensor of shape [batch_size]
        img_shape - Shape of the input images, representing [batch, channels, height, width]
    Outputs:
        bpd - The negative log likelihood in bits per dimension for the given image.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    bpd = elbo / (np.log(2) *np.prod(img_shape[1:]))
    #######################
    # END OF YOUR CODE    #
    #######################
    return bpd


@torch.no_grad()
def visualize_manifold(decoder, grid_size=20):
    """
    Visualize a manifold over a 2 dimensional latent space. The images in the manifold
    should represent the decoder's output means (not binarized samples of those).
    Inputs:
        decoder - Decoder model such as LinearDecoder or ConvolutionalDecoder.
        grid_size - Number of steps/images to have per axis in the manifold.
                    Overall you need to generate grid_size**2 images, and the distance
                    between different latents in percentiles is 1/grid_size
    Outputs:
        img_grid - Grid of images representing the manifold.
    """

    ## Hints:
    # - You can use the icdf method of the torch normal distribution  to obtain z values at percentiles.
    # - Use the range [0.5/grid_size, 1.5/grid_size, ..., (grid_size-0.5)/grid_size] for the percentiles.
    # - torch.meshgrid might be helpful for creating the grid of values
    # - You can use torchvision's function "make_grid" to combine the grid_size**2 images into a grid
    # - Remember to apply a softmax after the decoder

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    percentiles = torch.linspace(0.5/grid_size, (grid_size-0.5)/grid_size, grid_size)
    z_values = np.sqrt(2) * torch.erfinv(2*percentiles - 1)  # shape: (grid_size,)

    Z1, Z2 = torch.meshgrid(z_values, z_values, indexing='ij') # shape: (grid_size, grid_size)
    Z = torch.stack([Z1, Z2], dim=-1).view(-1, 2)  # Flatten to (grid_size^2, 2)

    logits = decoder(Z)

    probabilities = torch.nn.functional.softmax(logits, dim=1)
    num_bins = probabilities.shape[1]   
    values = torch.linspace(0, 1, num_bins)
    image = (probabilities * values[None, :, None, None]).sum(dim=1, keepdim=True)  # (G^2,1,H,W)

    img_grid = make_grid(image, nrow=grid_size, normalize=True, value_range=(0, 1), pad_value=0.5)    
    img_grid = img_grid.detach().cpu()
    #######################
    # END OF YOUR CODE    #
    #######################

    return img_grid

