################################################################################
# MIT License
#
# Copyright (c) 2024 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2024
# Date Created: 2024-10-28
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule, LinearModule
import cifar10_utils

import torch
import matplotlib.pyplot as plt


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions between 0 and 1,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # get the class that would be chosen in prediction by taking the one with highest probability
    predicted_classes = np.argmax(predictions, axis = 1)

    # make a new vector that checks with guesses are correct
    correct_predictions = (predicted_classes == targets)

    # take the mean out of the vector to see how many guesses are correct
    accuracy = np.mean(correct_predictions)

    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    accuracy_vector = []
    total_size = 0 
    for data in data_loader:
        #separate the data into input and desired output
        inputs, labels = data 

        #perform forward pass to obtain our model predictions
        #to take into account that the batch size might not be the same for all batches, 
        #we are going to save batch size to compute weigthted average
        prediction = model.forward(inputs)
        accuracy_vector.append(accuracy(prediction, labels)*labels.shape[0])
        total_size += labels.shape[0]


    avg_accuracy = np.sum(accuracy_vector)/(total_size)

    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that
                     performed best on the validation. Between 0.0 and 1.0
      logging_dict: An arbitrary object containing logging information. This is for you to
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model.
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set,
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size, return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Get a batch of data to determine the input shape
    x, y = next(iter(cifar10_loader['train']))

    # Flatten the input images and get the number of input features
    n_inputs = x.shape[1] * x.shape[2] * x.shape[3]   # 32 * 32 * 3 = 3072 np.prod(x)
    n_classes = len(cifar10["train"].dataset.classes) # 10                                    # CIFAR-10 has 10 classes

    # TODO: Initialize model and loss module
    model = MLP(n_inputs=n_inputs, n_hidden=hidden_dims, n_classes=n_classes)
    loss_module = CrossEntropyModule()

    # TODO: Training loop including validation
    # initialise the validation values and the best model search
    best_valid_acc = 0.0
    best_model = None
    logging_dict = {'train_loss': [], 'train_acc': [], 'valid_accuracies': []}

    for epoch in tqdm(range(epochs)):
      
        epoch_loss = 0.0
        train_hits = 0
        count = 0
        

        for step, data in enumerate(cifar10_loader['train']):
            input, label = data
            data_number= label.shape[0]
            # forward propagation
            prediction = model.forward(input.reshape(input.shape[0], -1))

            # Keep track of errors and iteration values for validation
            
            epoch_loss += loss_module.forward(prediction, label)*data_number 
            train_hits += (accuracy(prediction, label) * data_number)
            count += data_number

            # Gradient calculation and Backpropagation
            loss_gradient = loss_module.backward(prediction, label)
            model.backward(loss_gradient)

            # Update//learning
            for layer in model.modules:
                if isinstance(layer, LinearModule): #only update the linear layers
                    layer.params['weight'] -= lr * layer.grads['weight']
                    layer.params['bias'] -= lr * layer.grads['bias']
            # we clear the gradients and cache after each step
            model.clear_cache()

        logging_dict['train_loss'].append(round(epoch_loss / count, 3))
        logging_dict['train_acc'].append(round(train_hits / count, 3))
        
        # Validation
        valid_acc = evaluate_model(model, cifar10_loader['validation'])
        logging_dict['valid_accuracies'].append(valid_acc)
        
        # Saves best model during validation
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model = deepcopy(model)


    # TODO: Test best model
    test_accuracy = evaluate_model(best_model, cifar10_loader['test'])


    # TODO: Add any information you might want to save for plotting
    
    val_accuracies = logging_dict['valid_accuracies']

    model = best_model


    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    model, val_accuracies, test_accuracy, logging_dict = train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    print('test accuracy is: {test_accuracy}')
    def plot_loss_curve(logging_dict, val_accuracies):
        """
        Plots the loss and accuracy curves.
        Args:
            logging_dict: An arbitrary object containing logging information.
        """
        loss_curve = logging_dict['train_loss']

        acc_curve = val_accuracies
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(loss_curve)
        ax1.set_title('Loss curve during training')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')

        ax2.plot(acc_curve)
        ax2.set_title('Accuracy during validation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')

        plt.savefig("loss_curve_training_numpy.jpg")

    plot_loss_curve(logging_dict, val_accuracies)