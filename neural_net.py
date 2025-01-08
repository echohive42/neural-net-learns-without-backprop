import numpy as np
import random
from termcolor import colored
import time

# Constants
INPUT_A = 5
INPUT_B = 9
EXPECTED_OUTPUT = INPUT_A + INPUT_B
LEARNING_RATE = 0.1
MAX_ITERATIONS = 1000000
ERROR_THRESHOLD = 0.01

class SimpleNeuron:
    def __init__(self, num_inputs):
        try:
            print(colored(f"Initializing neuron with {num_inputs} inputs", "cyan"))
            # Random weights between -1 and 1
            self.weights = np.random.uniform(-1, 1, num_inputs)
            self.bias = np.random.uniform(-1, 1)
        except Exception as e:
            print(colored(f"Error initializing neuron: {str(e)}", "red"))
            raise

    def activate(self, inputs):
        try:
            # Simple activation function (ReLU)
            result = np.dot(inputs, self.weights) + self.bias
            return max(0, result)
        except Exception as e:
            print(colored(f"Error in activation: {str(e)}", "red"))
            raise

class SimpleNeuralNetwork:
    def __init__(self):
        try:
            print(colored("Initializing Neural Network...", "green"))
            self.neuron1 = SimpleNeuron(2)  # First neuron with 2 inputs
            self.neuron2 = SimpleNeuron(1)  # Second neuron with 1 input
        except Exception as e:
            print(colored(f"Error initializing network: {str(e)}", "red"))
            raise

    def forward(self, inputs):
        try:
            # First layer
            hidden_output = self.neuron1.activate(inputs)
            # Second layer
            final_output = self.neuron2.activate([hidden_output])
            return final_output
        except Exception as e:
            print(colored(f"Error in forward pass: {str(e)}", "red"))
            raise

    def random_adjust_weights(self):
        try:
            # Randomly adjust weights and biases
            self.neuron1.weights += np.random.uniform(-LEARNING_RATE, LEARNING_RATE, 2)
            self.neuron1.bias += random.uniform(-LEARNING_RATE, LEARNING_RATE)
            self.neuron2.weights += np.random.uniform(-LEARNING_RATE, LEARNING_RATE, 1)
            self.neuron2.bias += random.uniform(-LEARNING_RATE, LEARNING_RATE)
        except Exception as e:
            print(colored(f"Error adjusting weights: {str(e)}", "red"))
            raise

def main():
    try:
        print(colored("Starting Neural Network Training...", "green"))
        network = SimpleNeuralNetwork()
        
        inputs = np.array([INPUT_A, INPUT_B])
        best_error = float('inf')
        best_weights = None
        
        for i in range(MAX_ITERATIONS):
            output = network.forward(inputs)
            error = abs(output - EXPECTED_OUTPUT)
            
            # Print every iteration
            print(colored(f"Iteration {i}: Output = {output:.4f}, Error = {error:.4f}", "blue"))
            
            if error < best_error:
                best_error = error
                best_weights = {
                    'neuron1_weights': network.neuron1.weights.copy(),
                    'neuron1_bias': network.neuron1.bias,
                    'neuron2_weights': network.neuron2.weights.copy(),
                    'neuron2_bias': network.neuron2.bias
                }
                print(colored(f"New Best! Iteration {i}: Output = {output:.4f}, Error = {error:.4f}", "yellow"))
            
            if error < ERROR_THRESHOLD:
                print(colored(f"\nSuccess! Found solution with error {error:.4f}", "green"))
                break
                
            network.random_adjust_weights()
            
        print(colored("\nFinal Results:", "cyan"))
        print(colored(f"Best Error: {best_error:.4f}", "cyan"))
        print(colored(f"Expected Output: {EXPECTED_OUTPUT}", "cyan"))
        print(colored(f"Actual Output: {output:.4f}", "cyan"))
        
    except Exception as e:
        print(colored(f"Error in main execution: {str(e)}", "red"))
        raise

if __name__ == "__main__":
    main() 