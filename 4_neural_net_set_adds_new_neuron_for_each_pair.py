import numpy as np
import random
from termcolor import colored
import time
import json
import os

# Constants
TRAINING_PAIRS = {
    'pair1': {'a': 2, 'b': 5},
    'pair2': {'a': 3, 'b': 7},
    'pair3': {'a': 4, 'b': 4},
    'pair4': {'a': 1, 'b': 9},
    'pair5': {'a': 6, 'b': 6}
}
LEARNING_RATE = 0.1
MAX_ITERATIONS = 10000000
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
            self.hidden_neurons = [SimpleNeuron(2)]  # Start with one hidden neuron
            self.output_neuron = SimpleNeuron(1)  # Output neuron
            print(colored(f"Initial network structure: {len(self.hidden_neurons)} hidden neurons", "cyan"))
        except Exception as e:
            print(colored(f"Error initializing network: {str(e)}", "red"))
            raise

    def add_neuron(self):
        try:
            self.hidden_neurons.append(SimpleNeuron(2))
            self.output_neuron = SimpleNeuron(len(self.hidden_neurons))  # Recreate output neuron with new input size
            print(colored(f"Added new neuron. Network now has {len(self.hidden_neurons)} hidden neurons", "yellow"))
        except Exception as e:
            print(colored(f"Error adding neuron: {str(e)}", "red"))
            raise

    def forward(self, inputs, pair_name=None):
        try:
            # Process inputs through all hidden neurons
            hidden_outputs = [neuron.activate(inputs) for neuron in self.hidden_neurons]
            # Process hidden layer outputs through output neuron
            final_output = self.output_neuron.activate(hidden_outputs)
            return final_output
        except Exception as e:
            print(colored(f"Error in forward pass: {str(e)}", "red"))
            raise

    def random_adjust_weights(self, pair_name=None):
        try:
            # Adjust weights for all hidden neurons
            for neuron in self.hidden_neurons:
                neuron.weights += np.random.uniform(-LEARNING_RATE, LEARNING_RATE, 2)
                neuron.bias += random.uniform(-LEARNING_RATE, LEARNING_RATE)
            # Adjust weights for output neuron
            self.output_neuron.weights += np.random.uniform(-LEARNING_RATE, LEARNING_RATE, len(self.hidden_neurons))
            self.output_neuron.bias += random.uniform(-LEARNING_RATE, LEARNING_RATE)
        except Exception as e:
            print(colored(f"Error adjusting weights: {str(e)}", "red"))
            raise

    def load_weights(self, weights_data):
        try:
            # Load weights for hidden neurons
            for i, neuron_weights in enumerate(weights_data['hidden_neurons']):
                if i >= len(self.hidden_neurons):
                    self.hidden_neurons.append(SimpleNeuron(2))
                self.hidden_neurons[i].set_weights(
                    neuron_weights['weights'],
                    neuron_weights['bias']
                )
            # Load weights for output neuron
            self.output_neuron = SimpleNeuron(len(self.hidden_neurons))
            self.output_neuron.set_weights(
                weights_data['output_neuron']['weights'],
                weights_data['output_neuron']['bias']
            )
            print(colored("Weights loaded successfully!", "green"))
        except Exception as e:
            print(colored(f"Error loading weights: {str(e)}", "red"))
            raise

def save_weights(weights_data):
    try:
        # Convert numpy arrays to lists for JSON serialization
        weights_to_save = {
            'hidden_neurons': [
                {
                    'weights': neuron_data['weights'].tolist(),
                    'bias': float(neuron_data['bias'])
                }
                for neuron_data in weights_data['hidden_neurons']
            ],
            'output_neuron': {
                'weights': weights_data['output_neuron']['weights'].tolist(),
                'bias': float(weights_data['output_neuron']['bias'])
            }
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
        
        overall_best_error = float('inf')
        best_weights = None
        
        for pair_name, pair_data in TRAINING_PAIRS.items():
            print(colored(f"\nTraining on {pair_name}: {pair_data['a']} + {pair_data['b']}", "cyan"))
            inputs = np.array([pair_data['a'], pair_data['b']])
            expected_output = pair_data['a'] + pair_data['b']
            best_error_for_pair = float('inf')
            
            for i in range(MAX_ITERATIONS):
                output = network.forward(inputs)
                error = abs(output - expected_output)
                
                if i % 1000 == 0:  # Print every 1000 iterations
                    print(colored(f"Iteration {i}: Output = {output:.4f}, Error = {error:.4f}", "blue"))
                
                if error < best_error_for_pair:
                    best_error_for_pair = error
                    if error < overall_best_error:
                        overall_best_error = error
                        best_weights = {
                            'hidden_neurons': [
                                {
                                    'weights': neuron.weights.copy(),
                                    'bias': neuron.bias
                                }
                                for neuron in network.hidden_neurons
                            ],
                            'output_neuron': {
                                'weights': network.output_neuron.weights.copy(),
                                'bias': network.output_neuron.bias
                            }
                        }
                        print(colored(f"New Best! Pair: {pair_name}, Output = {output:.4f}, Error = {error:.4f}", "yellow"))
                
                if error < ERROR_THRESHOLD:
                    success_msg = f"\nSuccess with {pair_name}!\n"
                    success_msg += f"Input: {pair_data['a']} + {pair_data['b']} = {expected_output}\n"
                    success_msg += f"Network Output: {output:.4f}\n"
                    success_msg += f"Error: {error:.4f}\n"
                    success_msg += f"Current network size: {len(network.hidden_neurons)} hidden neurons"
                    print(colored(success_msg, "green"))
                    
                    # Add a new neuron after success
                    network.add_neuron()
                    print(colored("Press Enter to continue with expanded network...", "yellow"))
                    input()
                    break
                    
                network.random_adjust_weights()
            
            print(colored(f"Best error for {pair_name}: {best_error_for_pair:.4f}", "cyan"))
        
        print(colored("\nTraining Complete for all pairs!", "cyan"))
        print(colored(f"Overall Best Error: {overall_best_error:.4f}", "cyan"))
        print(colored(f"Final network size: {len(network.hidden_neurons)} hidden neurons", "cyan"))
        
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
            print(colored("Options:", "yellow"))
            print(colored("1. Test with training pairs", "yellow"))
            print(colored("2. Test with custom numbers", "yellow"))
            print(colored("3. Back to main menu", "yellow"))
            
            choice = input(colored("\nEnter your choice (1-3): ", "green"))
            
            if choice == "1":
                print(colored("\nTesting with training pairs:", "cyan"))
                for pair_name, pair_data in TRAINING_PAIRS.items():
                    result = network.forward(np.array([pair_data['a'], pair_data['b']]))
                    expected = pair_data['a'] + pair_data['b']
                    error = abs(result - expected)
                    
                    print(colored(f"\n{pair_name}:", "yellow"))
                    print(colored(f"Input: {pair_data['a']} + {pair_data['b']}", "yellow"))
                    print(colored(f"Expected Output: {expected}", "yellow"))
                    print(colored(f"Neural Network Output: {result:.4f}", "yellow"))
                    print(colored(f"Error: {error:.4f}", "yellow"))
                
            elif choice == "2":
                print(colored("Enter two numbers to test:", "yellow"))
                try:
                    a = float(input(colored("Enter first number: ", "green")))
                    b = float(input(colored("Enter second number: ", "green")))
                    
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
            
            elif choice == "3":
                break
            else:
                print(colored("Invalid choice. Please try again.", "red"))
                
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