import numpy as np
import random
from termcolor import colored
import time
import json
import os

# Constants
INPUT_A = 2
INPUT_B = 5
EXPECTED_OUTPUT = INPUT_A + INPUT_B
LEARNING_RATE = 0.1
MAX_ITERATIONS = 1000000
ERROR_THRESHOLD = 0.01
WEIGHTS_FILE = "best_weights.json"

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

    def set_weights(self, weights, bias):
        try:
            self.weights = np.array(weights)
            self.bias = bias
        except Exception as e:
            print(colored(f"Error setting weights: {str(e)}", "red"))
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

    def load_weights(self, weights_data):
        try:
            self.neuron1.set_weights(
                weights_data['neuron1_weights'],
                weights_data['neuron1_bias']
            )
            self.neuron2.set_weights(
                weights_data['neuron2_weights'],
                weights_data['neuron2_bias']
            )
            print(colored("Weights loaded successfully!", "green"))
        except Exception as e:
            print(colored(f"Error loading weights: {str(e)}", "red"))
            raise

def save_weights(weights_data):
    try:
        # Convert numpy arrays to lists for JSON serialization
        weights_to_save = {
            'neuron1_weights': weights_data['neuron1_weights'].tolist(),
            'neuron1_bias': float(weights_data['neuron1_bias']),
            'neuron2_weights': weights_data['neuron2_weights'].tolist(),
            'neuron2_bias': float(weights_data['neuron2_bias'])
        }
        with open(WEIGHTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(weights_to_save, f, indent=4)
        print(colored(f"Weights saved to {WEIGHTS_FILE}", "green"))
    except Exception as e:
        print(colored(f"Error saving weights: {str(e)}", "red"))
        raise

def load_saved_weights():
    try:
        if os.path.exists(WEIGHTS_FILE):
            with open(WEIGHTS_FILE, 'r', encoding='utf-8') as f:
                weights_data = json.load(f)
            return weights_data
        return None
    except Exception as e:
        print(colored(f"Error loading weights file: {str(e)}", "red"))
        return None

def train_network():
    try:
        print(colored("Starting Neural Network Training...", "green"))
        network = SimpleNeuralNetwork()
        
        inputs = np.array([INPUT_A, INPUT_B])
        best_error = float('inf')
        best_weights = None
        
        for i in range(MAX_ITERATIONS):
            output = network.forward(inputs)
            error = abs(output - EXPECTED_OUTPUT)
            
            if i % 1000 == 0:  # Print every 1000 iterations
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
        
        print(colored("\nTraining Complete!", "cyan"))
        print(colored(f"Best Error: {best_error:.4f}", "cyan"))
        
        # Save the best weights
        if best_weights:
            save_weights(best_weights)
        
        return best_weights
    except Exception as e:
        print(colored(f"Error in training: {str(e)}", "red"))
        raise

def test_network(weights_data=None):
    try:
        network = SimpleNeuralNetwork()
        if weights_data:
            network.load_weights(weights_data)
        
        while True:
            print(colored("\n=== Neural Network Testing Interface ===", "cyan"))
            print(colored("Enter two numbers to test (or 'q' to quit):", "yellow"))
            
            user_input = input(colored("Enter first number: ", "green"))
            if user_input.lower() == 'q':
                break
                
            try:
                a = float(user_input)
                b = float(input(colored("Enter second number: ", "green")))
                
                # Process the input
                result = network.forward(np.array([a, b]))
                expected = a + b
                error = abs(result - expected)
                
                print(colored("\nResults:", "cyan"))
                print(colored(f"Input: {a} + {b}", "yellow"))
                print(colored(f"Expected Output: {expected}", "yellow"))
                print(colored(f"Neural Network Output: {result:.4f}", "yellow"))
                print(colored(f"Error: {error:.4f}", "yellow"))
                
            except ValueError:
                print(colored("Please enter valid numbers!", "red"))
                
    except Exception as e:
        print(colored(f"Error in testing: {str(e)}", "red"))
        raise

def main():
    try:
        while True:
            print(colored("\n=== Neural Network Menu ===", "cyan"))
            print(colored("1. Train New Network", "yellow"))
            print(colored("2. Test Network", "yellow"))
            print(colored("3. Exit", "yellow"))
            
            choice = input(colored("\nEnter your choice (1-3): ", "green"))
            
            if choice == "1":
                weights = train_network()
            elif choice == "2":
                weights = load_saved_weights()
                if weights:
                    test_network(weights)
                else:
                    print(colored("No saved weights found. Training new network...", "red"))
                    weights = train_network()
                    test_network(weights)
            elif choice == "3":
                print(colored("Goodbye!", "cyan"))
                break
            else:
                print(colored("Invalid choice. Please try again.", "red"))
                
    except Exception as e:
        print(colored(f"Error in main execution: {str(e)}", "red"))
        raise

if __name__ == "__main__":
    main() 