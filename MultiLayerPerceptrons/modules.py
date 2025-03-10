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
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization.
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        # Note: For the sake of this assignment, please store the parameters
        # and gradients in this format, otherwise some unit tests might fail.
        self.params = {'weight': None, 'bias': None} # Model parameters
        self.grads = {'weight': None, 'bias': None} # Gradients

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        # The first layer comes from random inputs, not ReLU output's, so it has to be initialised following a Normal w~N(0,1/d)
        # put sqrt() because np.random.norm asks standard deviation and not variance
        self.input_layer = input_layer
        self.in_features = in_features
        self.out_features = out_features

        if input_layer:
            self.params['weight'] = np.random.normal(0, 1/np.sqrt(in_features), (in_features, out_features))

        # Kaiming comes into factor on layers that have ReLU outputs, drawing weights w~N(0,2/d)
        else:
            self.params['weight'] = np.random.normal(0, np.sqrt(2/in_features), (in_features, out_features))

        # bias and gradients can be initialised at zero
        self.params['bias'] = np.zeros(out_features)
        self.grads['weight'] = np.zeros((in_features, out_features))
        self.grads['bias'] = np.zeros(out_features)
        self.out = None
        self.input = None

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        out = np.matmul(x , self.params['weight']) + self.params['bias']
        self.input = x      #Safe the value for backpropagation
        self.out = out

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################


        # Following our calculations, dw= x^T * dL/dY 
        self.grads['weight'] = np.matmul(self.input.T , dout)

        # at first we had self.grads['bias'] = np.ones(dout.shape[0]).T @ dout , whihc is vectorial, but less efficient

        self.grads['bias'] = np.sum(dout, axis=0)
        dx = np.matmul(dout , self.params['weight'].T)



        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.input = None
        self.out = None
        #######################
        # END OF YOUR CODE    #
        #######################


class ELUModule(object):
    """
    ELU activation module.
    """

    def __init__(self, alpha = 1):
        self.alpha = alpha

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        self.input = x           #Safe the value for backpropagation
        out = np.where( x > 0, x, self.alpha*(np.exp(x)-1))

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        dx =  dout * np.where(self.input > 0, 1, self.alpha*np.exp(self.input))

        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        self.input = None

        #######################
        # END OF YOUR CODE    #
        #######################


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        
        b = np.max(x, axis = 1, keepdims = True)
        y = np.exp(x - b)
        out = (y / y.sum(axis = 1, keepdims = True))
        self.out=out

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # See that Y= self.out, dL/dY= dout and  1 @ 1.T = a matrix of ones
        
        size = dout.shape[1]
        dx = self.out * (dout - np.matmul((dout * self.out), np.ones((size, size))))

        #######################
        # END OF YOUR CODE    #
        #######################

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        self.out = None

        #######################
        # END OF YOUR CODE    #
        #######################


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # We reshape y to assure that it has the right dimension, we put an if condition to avoid reshaping when it is not needed
        y = np.eye(x.shape[1])[y]
        epsilon = 1e-12  #we add a small value to never get into computing log of 0
        out =  (np.sum(-y * np.log(x), axis = 1)).mean()

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        y = np.eye(x.shape[1])[y]
        dx= (-1 / x.shape[0] * np.divide(y,x))


        #######################
        # END OF YOUR CODE    #
        #######################

        return dx