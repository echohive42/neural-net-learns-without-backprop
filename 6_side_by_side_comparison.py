import numpy as np
import random
from termcolor import colored
import time
import json
import os
import sys
from importlib.util import spec_from_file_location, module_from_spec

# Import the neural network modules
fixed_net_path = os.path.join(os.path.dirname(__file__), "3_neural_net_train_and_test_set_of_numbers.py")
dynamic_net_path = os.path.join(os.path.dirname(__file__), "4_neural_net_set_adds_new_neuron_for_each_pair.py")

# Load fixed network module
fixed_net_spec = spec_from_file_location("fixed_net", fixed_net_path)
fixed_net = module_from_spec(fixed_net_spec)
fixed_net_spec.loader.exec_module(fixed_net)

# Load dynamic network module
dynamic_net_spec = spec_from_file_location("dynamic_net", dynamic_net_path)
dynamic_net = module_from_spec(dynamic_net_spec)
dynamic_net_spec.loader.exec_module(dynamic_net)

# Training pairs for initial training
TRAINING_PAIRS = {
    'pair1': {'a': 2, 'b': 2},
    'pair2': {'a': 3, 'b': 4},
    'pair3': {'a': 1, 'b': 3},
    'pair4': {'a': 4, 'b': 9},
    'pair5': {'a': 7, 'b': 8}
}

def train_networks():
    """Train both networks on the training pairs"""
    try:
        print(colored("\n=== Training Fixed Network ===", "cyan", attrs=['bold']))
        fixed_network = fixed_net.SimpleNeuralNetwork()
        
        print(colored("\n=== Training Dynamic Network ===", "cyan", attrs=['bold']))
        dynamic_network = dynamic_net.SimpleNeuralNetwork()
        
        # Train both networks on the same pairs
        for pair_name, pair_data in TRAINING_PAIRS.items():
            print(colored(f"\nTraining both networks on {pair_name}: {pair_data['a']} + {pair_data['b']} = {pair_data['a'] + pair_data['b']}", "yellow"))
            inputs = np.array([pair_data['a'], pair_data['b']])
            expected = pair_data['a'] + pair_data['b']
            
            # Train Fixed Network
            iterations = 0
            best_error_fixed = float('inf')
            while iterations < fixed_net.MAX_ITERATIONS:
                iterations += 1
                output = fixed_network.forward(inputs)
                error = abs(output - expected)
                
                if error < best_error_fixed:
                    best_error_fixed = error
                
                if error < fixed_net.ERROR_THRESHOLD:
                    break
                    
                fixed_network.random_adjust_weights()
            
            print(colored(f"Fixed Network - Iterations: {iterations}, Final Error: {best_error_fixed:.4f}", "blue"))
            
            # Train Dynamic Network
            iterations = 0
            best_error_dynamic = float('inf')
            while iterations < dynamic_net.MAX_ITERATIONS:
                iterations += 1
                output = dynamic_network.forward(inputs, pair_name)
                error = abs(output - expected)
                
                if error < best_error_dynamic:
                    best_error_dynamic = error
                
                if error < dynamic_net.ERROR_THRESHOLD:
                    dynamic_network.add_neuron()  # Add new neuron after success
                    break
                    
                dynamic_network.random_adjust_weights(pair_name)
            
            print(colored(f"Dynamic Network - Iterations: {iterations}, Final Error: {best_error_dynamic:.4f}", "magenta"))
        
        return fixed_network, dynamic_network
    
    except Exception as e:
        print(colored(f"Error in training: {str(e)}", "red"))
        raise

def side_by_side_test(fixed_network, dynamic_network):
    """Interactive testing interface for both networks"""
    try:
        while True:
            print(colored("\n=== Side by Side Network Testing ===", "cyan", attrs=['bold']))
            print(colored("Enter two numbers to test (or 'q' to quit):", "yellow"))
            
            # Get input
            try:
                first = input(colored("Enter first number: ", "green"))
                if first.lower() == 'q':
                    break
                    
                a = float(first)
                b = float(input(colored("Enter second number: ", "green")))
                
                # Prepare input and expected output
                inputs = np.array([a, b])
                expected = a + b
                
                # Test both networks
                print(colored("\nResults:", "cyan"))
                print("=" * 60)
                print(colored(f"Input: {a} + {b} = {expected}", "yellow"))
                print("=" * 60)
                
                # Fixed Network
                fixed_output = fixed_network.forward(inputs)
                fixed_error = abs(fixed_output - expected)
                print(colored("Fixed Network:", "blue", attrs=['bold']))
                print(f"Output: {fixed_output:.4f}")
                print(f"Error:  {fixed_error:.4f}")
                
                print("-" * 30)
                
                # Dynamic Network
                dynamic_output = dynamic_network.forward(inputs)
                dynamic_error = abs(dynamic_output - expected)
                print(colored("Dynamic Network:", "magenta", attrs=['bold']))
                print(f"Output: {dynamic_output:.4f}")
                print(f"Error:  {dynamic_error:.4f}")
                
                print("=" * 60)
                
                # Compare results
                if fixed_error < dynamic_error:
                    print(colored("Fixed Network performed better!", "blue", attrs=['bold']))
                elif dynamic_error < fixed_error:
                    print(colored("Dynamic Network performed better!", "magenta", attrs=['bold']))
                else:
                    print(colored("Both networks performed equally!", "yellow", attrs=['bold']))
                
            except ValueError:
                print(colored("Please enter valid numbers!", "red"))
                continue
                
    except Exception as e:
        print(colored(f"Error in testing: {str(e)}", "red"))
        raise

def main():
    try:
        print(colored("Training networks on initial pairs...", "cyan"))
        fixed_network, dynamic_network = train_networks()
        
        print(colored("\nStarting side by side testing...", "cyan"))
        side_by_side_test(fixed_network, dynamic_network)
        
    except Exception as e:
        print(colored(f"Error in main execution: {str(e)}", "red"))
        raise

if __name__ == "__main__":
    main() 