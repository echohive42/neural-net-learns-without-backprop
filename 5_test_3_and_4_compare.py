import numpy as np
import random
from termcolor import colored
import time
import json
import os
import sys
from typing import Dict, Any, Tuple
import statistics
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

# Test Constants
NUM_TEST_RUNS = 3  # Increased number of test runs for better statistics
TEST_PAIRS = {
    'train_pairs': {
        'pair1': {'a': 2, 'b': 2},
        'pair2': {'a': 3, 'b': 4},
        'pair3': {'a': 1, 'b': 3},
        'pair4': {'a': 4, 'b': 9},
        'pair5': {'a': 7, 'b': 8}
    },
    'validation_pairs': {  # New pairs to test generalization
        'pair6': {'a': 5, 'b': 5},   # Sum: 10
        'pair7': {'a': 2, 'b': 4},   # Sum: 6
        'pair8': {'a': 8, 'b': 3},   # Sum: 11
        'pair9': {'a': 7, 'b': 6},   # Sum: 13
        'pair10': {'a': 1, 'b': 1},  # Sum: 2 (edge case)
        'pair11': {'a': 9, 'b': 9},  # Sum: 18 (edge case)
        'pair12': {'a': 0, 'b': 5},   # Sum: 5 (edge case with zero)
        'pair13': {'a': 14, 'b': 21},  # Sum: 10
    }
}

class NetworkTester:
    def __init__(self):
        self.results = {
            'fixed_network': [],
            'dynamic_network': []
        }

    def run_single_test(self, network_type: str, network_module: Any) -> Dict[str, Any]:
        """Run a single complete test of the network"""
        try:
            start_time = time.time()
            
            # Training phase
            print(colored(f"\nTraining {network_type}...", "cyan"))
            training_start = time.time()
            network = network_module.SimpleNeuralNetwork()
            
            iterations_per_pair = {}
            errors_per_pair = {}
            convergence_times = {}
            
            for pair_name, pair_data in TEST_PAIRS['train_pairs'].items():
                pair_start_time = time.time()
                inputs = np.array([pair_data['a'], pair_data['b']])
                expected = pair_data['a'] + pair_data['b']
                
                iterations = 0
                best_error = float('inf')
                
                while iterations < network_module.MAX_ITERATIONS:
                    iterations += 1
                    output = network.forward(inputs)
                    error = abs(output - expected)
                    
                    if error < best_error:
                        best_error = error
                    
                    if error < network_module.ERROR_THRESHOLD:
                        break
                    
                    if hasattr(network, 'random_adjust_weights'):
                        if network_type == 'Dynamic Network':
                            network.random_adjust_weights(pair_name)
                        else:
                            network.random_adjust_weights()
                
                convergence_times[pair_name] = time.time() - pair_start_time
                iterations_per_pair[pair_name] = iterations
                errors_per_pair[pair_name] = best_error
                
                print(colored(f"{pair_name}: {iterations} iterations, time: {convergence_times[pair_name]:.2f}s, final error: {best_error:.4f}", "yellow"))
            
            training_time = time.time() - training_start
            
            # Validation phase
            print(colored("\nTesting on validation pairs...", "cyan"))
            validation_errors = {}
            validation_times = {}
            
            for pair_name, pair_data in TEST_PAIRS['validation_pairs'].items():
                pair_start_time = time.time()
                inputs = np.array([pair_data['a'], pair_data['b']])
                expected = pair_data['a'] + pair_data['b']
                
                if network_type == 'Dynamic Network':
                    output = network.forward(inputs, None)
                else:
                    output = network.forward(inputs)
                    
                error = abs(output - expected)
                validation_errors[pair_name] = error
                validation_times[pair_name] = time.time() - pair_start_time
                
                print(colored(f"{pair_name}: {pair_data['a']} + {pair_data['b']} = {expected}, Output: {output:.4f}, Error: {error:.4f}, Time: {validation_times[pair_name]:.4f}s", "yellow"))
            
            total_time = time.time() - start_time
            
            return {
                'training_time': training_time,
                'total_time': total_time,
                'iterations_per_pair': iterations_per_pair,
                'training_errors': errors_per_pair,
                'validation_errors': validation_errors,
                'convergence_times': convergence_times,
                'validation_times': validation_times,
                'avg_training_error': statistics.mean(errors_per_pair.values()),
                'avg_validation_error': statistics.mean(validation_errors.values()),
                'max_iterations': max(iterations_per_pair.values()),
                'min_iterations': min(iterations_per_pair.values()),
                'avg_iterations': statistics.mean(iterations_per_pair.values()),
                'avg_convergence_time': statistics.mean(convergence_times.values()),
                'avg_validation_time': statistics.mean(validation_times.values())
            }
            
        except Exception as e:
            print(colored(f"Error in {network_type} test: {str(e)}", "red"))
            raise

    def run_comparison(self):
        """Run multiple tests and compile statistics"""
        print(colored("\n=== Starting Neural Network Comparison ===", "cyan"))
        
        for run in range(NUM_TEST_RUNS):
            print(colored(f"\nTest Run {run + 1}/{NUM_TEST_RUNS}", "green", attrs=['bold']))
            
            # Test fixed network
            print(colored("\nTesting Fixed Network (3_neural_net_train_and_test_set_of_numbers.py)", "cyan"))
            self.results['fixed_network'].append(
                self.run_single_test('Fixed Network', fixed_net)
            )
            
            # Test dynamic network
            print(colored("\nTesting Dynamic Network (4_neural_net_set_adds_new_neuron_for_each_pair.py)", "cyan"))
            self.results['dynamic_network'].append(
                self.run_single_test('Dynamic Network', dynamic_net)
            )
        
        self.print_comparison_results()

    def print_comparison_results(self):
        """Print comprehensive comparison results"""
        print(colored("\n=== Neural Network Comparison Results ===", "green", attrs=['bold']))
        
        metrics = {
            'training_time': ('Average Training Time', lambda x: f"{x:.2f}s"),
            'avg_training_error': ('Average Training Error', lambda x: f"{x:.4f}"),
            'avg_validation_error': ('Average Validation Error', lambda x: f"{x:.4f}"),
            'avg_iterations': ('Average Iterations per Pair', lambda x: f"{x:.0f}"),
            'max_iterations': ('Maximum Iterations for any Pair', lambda x: f"{x:.0f}"),
            'avg_convergence_time': ('Average Convergence Time per Pair', lambda x: f"{x:.2f}s"),
            'avg_validation_time': ('Average Validation Time per Pair', lambda x: f"{x:.4f}s")
        }
        
        for network_type in ['fixed_network', 'dynamic_network']:
            print(colored(f"\n{network_type.replace('_', ' ').title()} Statistics:", "cyan", attrs=['bold']))
            
            for metric, (label, formatter) in metrics.items():
                values = [run[metric] for run in self.results[network_type]]
                avg_value = statistics.mean(values)
                std_dev = statistics.stdev(values) if len(values) > 1 else 0
                min_value = min(values)
                max_value = max(values)
                
                print(colored(f"\n{label}:", "yellow"))
                print(f"  Mean: {formatter(avg_value)}")
                print(f"  Std Dev: {formatter(std_dev)}")
                print(f"  Min: {formatter(min_value)}")
                print(f"  Max: {formatter(max_value)}")
        
        # Print detailed conclusion
        print(colored("\nDetailed Comparison:", "green", attrs=['bold']))
        
        # Training Performance
        fixed_train_error = statistics.mean([run['avg_training_error'] for run in self.results['fixed_network']])
        dynamic_train_error = statistics.mean([run['avg_training_error'] for run in self.results['dynamic_network']])
        print(colored("\nTraining Error Comparison:", "yellow"))
        print(f"Fixed Network: {fixed_train_error:.4f}")
        print(f"Dynamic Network: {dynamic_train_error:.4f}")
        print(f"Difference: {abs(fixed_train_error - dynamic_train_error):.4f}")
        
        # Validation Performance
        fixed_val_error = statistics.mean([run['avg_validation_error'] for run in self.results['fixed_network']])
        dynamic_val_error = statistics.mean([run['avg_validation_error'] for run in self.results['dynamic_network']])
        print(colored("\nValidation Error Comparison:", "yellow"))
        print(f"Fixed Network: {fixed_val_error:.4f}")
        print(f"Dynamic Network: {dynamic_val_error:.4f}")
        print(f"Difference: {abs(fixed_val_error - dynamic_val_error):.4f}")
        
        # Speed Comparison
        fixed_time = statistics.mean([run['training_time'] for run in self.results['fixed_network']])
        dynamic_time = statistics.mean([run['training_time'] for run in self.results['dynamic_network']])
        print(colored("\nTraining Time Comparison:", "yellow"))
        print(f"Fixed Network: {fixed_time:.2f}s")
        print(f"Dynamic Network: {dynamic_time:.2f}s")
        print(f"Difference: {abs(fixed_time - dynamic_time):.2f}s")
        
        # Overall Conclusion
        print(colored("\nConclusion:", "green", attrs=['bold']))
        
        # Training Error Conclusion
        if fixed_train_error < dynamic_train_error:
            print("- Fixed Network achieved lower training error")
        else:
            print("- Dynamic Network achieved lower training error")
        
        # Validation Error Conclusion
        if fixed_val_error < dynamic_val_error:
            print("- Fixed Network showed better generalization on validation data")
        else:
            print("- Dynamic Network showed better generalization on validation data")
        
        # Speed Conclusion
        if fixed_time < dynamic_time:
            print("- Fixed Network trained faster")
        else:
            print("- Dynamic Network trained faster")
        
        # Print recommendations
        print(colored("\nRecommendations:", "yellow"))
        if fixed_val_error < dynamic_val_error and fixed_time < dynamic_time:
            print("The Fixed Network appears to be the better choice overall, offering both better accuracy and faster training.")
        elif dynamic_val_error < fixed_val_error and dynamic_time < fixed_time:
            print("The Dynamic Network appears to be the better choice overall, offering both better accuracy and faster training.")
        else:
            if fixed_val_error < dynamic_val_error:
                print("Choose Fixed Network if accuracy is the priority.")
            else:
                print("Choose Dynamic Network if accuracy is the priority.")
            if fixed_time < dynamic_time:
                print("Choose Fixed Network if training speed is the priority.")
            else:
                print("Choose Dynamic Network if training speed is the priority.")

def main():
    try:
        tester = NetworkTester()
        tester.run_comparison()
    except Exception as e:
        print(colored(f"Error in comparison test: {str(e)}", "red"))
        raise

if __name__ == "__main__":
    main()
