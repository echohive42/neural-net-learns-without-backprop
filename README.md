# Simple Neural Network Without Backpropagation

This project implements basic neural networks from scratch that learn through random weight adjustments rather than traditional backpropagation. The networks attempt to learn a simple addition operation.

## Features

- Built from scratch using only NumPy
- No backpropagation - uses random weight adjustments
- Colored terminal output showing training progress
- Error tracking and best weights saving
- Comprehensive error handling
- Multiple implementations for learning purposes
- Comparative analysis tools

## ❤️ Support & Get 400+ AI Projects

This is one of 400+ fascinating projects in my collection! [Support me on Patreon](https://www.patreon.com/c/echohive42/membership) to get:

- 🎯 Access to 400+ AI projects (and growing daily!)
  - Including advanced projects like [2 Agent Real-time voice template with turn taking](https://www.patreon.com/posts/2-agent-real-you-118330397)
- 📥 Full source code & detailed explanations
- 📚 1000x Cursor Course
- 🎓 Live coding sessions & AMAs
- 💬 1-on-1 consultations (higher tiers)
- 🎁 Exclusive discounts on AI tools & platforms (up to $180 value)

## Overview

The project contains multiple implementations with increasing complexity:

### 1. Basic Implementation (1_neural_net_without_backprop.py)

- 2 neurons arranged in 2 layers
- First neuron takes 2 inputs (a=5, b=9)
- Second neuron takes the output from first neuron
- Goal is to learn the addition operation (a + b = 14)
- Uses ReLU activation function
- Random weight adjustment learning strategy
- Real-time training visualization in terminal

### 2. Advanced Implementation (2_neural_net_train_and_test.py)

- Same neural network architecture as basic version
- Added features:
  - Weight saving/loading functionality
  - Interactive testing interface
  - User-friendly terminal UI with menu system
  - Ability to test custom input values
  - Progress tracking with reduced output clutter

### 3. Multi-Pair Training (3_neural_net_train_and_test_set_of_numbers.py)

- Fixed architecture with 2 neurons
- Trains on multiple pairs of numbers
- Maintains consistent network structure
- Tests generalization across different number pairs
- Enhanced error tracking and weight management

### 4. Dynamic Growing Network (4_neural_net_set_adds_new_neuron_for_each_pair.py)

- Starts with a single hidden neuron
- Dynamically adds new neurons after learning each pair
- Adaptive architecture that grows with training
- Specialized neurons for different number patterns
- Advanced weight management for variable network size

### 5. Network Comparison Tool (5_test_3_and_4_compare.py)

- Comprehensive testing framework
- Compares fixed vs dynamic architectures
- Measures:
  - Training speed
  - Training accuracy
  - Generalization ability
  - Convergence time
  - Overall effectiveness
- Detailed statistics and performance analysis
- Validation testing on new number pairs
