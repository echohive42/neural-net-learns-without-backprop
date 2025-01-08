# Simple Neural Network Without Backpropagation

This project implements a basic neural network from scratch that learns through random weight adjustments rather than traditional backpropagation. The network attempts to learn a simple addition operation.

## Overview

The neural network consists of:
- 2 neurons arranged in 2 layers
- First neuron takes 2 inputs (a=5, b=9)
- Second neuron takes the output from first neuron
- Goal is to learn the addition operation (a + b = 14)
- Uses ReLU activation function
- Random weight adjustment learning strategy

## Features

- Built from scratch using only NumPy
- No backpropagation - uses random weight adjustments
- Colored terminal output showing training progress
- Error tracking and best weights saving
- Comprehensive error handling

## Requirements 