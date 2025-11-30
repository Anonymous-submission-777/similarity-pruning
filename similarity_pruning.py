import numpy as np
from Functions import (
    txt_loader,
    load_similarity_statistics,
    KNN_DTW_original,
    KNN_DTW_with_pruning,
    save_pruning_results,
)
import time

from collections import defaultdict

from collections import defaultdict

def print_results(results):
    """Print Results"""
    print("\n" + "="*60)
    print(f"Dataset: {results['dataset_name']}")
    print(f"Method: {results['method']}")
    print(f"Parameters: w={results['w']}, k={results['k']}")
    
    if results['method'] == 'Pruning':
        print(f"Same Class Distance Median: {results['same_class_median']:.6f}")
        print(f"Pruning Coefficient: {results['pruning_coefficient']}")
        print(f"Pruning Threshold: {results['pruning_threshold']:.6f}")
        print(f"Early Stop Threshold: {results['early_stop_count']} distances")
    
    print("="*60)
    
    print(f"\nClassification Results:")
    print(f"  Accuracy: {results['accuracy']:.6f} ({results['accuracy']*100:.2f}%)")
    print(f"  Total Time: {results['total_time']:.2f} seconds")
    
    if results['method'] == 'Pruning':
        print(f"  Pruning Rate: {results['pruning_rate']:.6f} ({results['pruning_rate']*100:.2f}%)")
        print(f"  Speedup Factor: {results['speedup_factor']:.2f}x")
        
        if results['pruning_rate'] > 0:
            print(f"\nPruning Effectiveness Analysis:")
            print(f"  • {results['pruning_rate']*100:.1f}% of test samples triggered early stopping")
            print(f"  • Average Speedup: {results['speedup_factor']:.2f}x")
    
    print("="*60)


def estimate_baseline_time(dataset_name, w, k):
    """
    Estimate baseline time (full KNN-DTW time without pruning)
    Using a simplified estimation method here.
    """
    # Load data to get scale information
    train_data = txt_loader(f'C:\\Users\\luoyu\\Desktop\\Univariate_arff\\{dataset_name}\\{dataset_name}_TRAIN.txt')
    test_data = txt_loader(f'C:\\Users\\luoyu\\Desktop\\Univariate_arff\\{dataset_name}\\{dataset_name}_TEST.txt')
    
    # Estimate baseline time based on data scale (this is a rough estimate)
    train_size = len(train_data)
    test_size = len(test_data)
    
    # Assume each CDTW calculation takes 0.001 seconds (adjust this value based on reality)
    estimated_cdtw_time = 0.001
    baseline_time = train_size * test_size * estimated_cdtw_time
    
    return baseline_time

def main():
    """Main Function"""
    
    # Dataset Configuration (Dataset Name, w parameter, k parameter)
    datasets = [
        ("Car", 6, 1)
    ]
    
    # Pruning Parameters Configuration
    PRUNING_COEFFICIENT = 0.05  # Pruning threshold coefficient, adjustable
    EARLY_STOP_COUNT = 2     # Minimum distance count for early stopping, adjustable
    
    print("Starting KNN-DTW Experiment...")
    print("Comparing performance of Original KNN-DTW vs Pruning KNN-DTW")
    print(f"Pruning Parameters: Coefficient={PRUNING_COEFFICIENT}, Early Stop Threshold={EARLY_STOP_COUNT}")
    
    for dataset_name, w, k in datasets:
        try:
            print(f"\n{'='*60}")
            print(f"Processing Dataset: {dataset_name}")
            print(f"{'='*60}")
            
            # 1. Run Original KNN-DTW Classification
            print("\n1. Running Original KNN-DTW Classification...")
            original_accuracy, original_time = KNN_DTW_original(dataset_name, w, k)
            
            # Organize original results
            original_results = {
                'dataset_name': dataset_name,
                'method': 'Original',
                'w': w,
                'k': k,
                'accuracy': original_accuracy,
                'total_time': original_time
            }
            
            # Print original results
            print_results(original_results)
            
            # Save original results
            save_pruning_results(original_results)
            
            # 2. Run Pruning KNN-DTW Classification
            print("\n2. Running Pruning KNN-DTW Classification...")
            
            # Load same class distance median
            same_class_median = load_similarity_statistics(dataset_name, w)
            if same_class_median is None:
                print(f"Skipping pruning experiment: Unable to retrieve statistics")
                continue
            
            # Run KNN-DTW with pruning
            pruning_accuracy, pruning_time, pruning_rate = KNN_DTW_with_pruning(
                dataset_name, w, k, same_class_median, pruning_coefficient=PRUNING_COEFFICIENT, early_stop_count=EARLY_STOP_COUNT
            )
            
            # Calculate speedup factor (relative to original method)
            speedup_factor = original_time / pruning_time if pruning_time > 0 else 1.0
            
            # Organize pruning results
            pruning_results = {
                'dataset_name': dataset_name,
                'method': 'Pruning',
                'w': w,
                'k': k,
                'same_class_median': same_class_median,
                'pruning_coefficient': PRUNING_COEFFICIENT,
                'pruning_threshold': PRUNING_COEFFICIENT * same_class_median,
                'early_stop_count': EARLY_STOP_COUNT,
                'accuracy': pruning_accuracy,
                'total_time': pruning_time,
                'pruning_rate': pruning_rate,
                'speedup_factor': speedup_factor
            }
            
            # Print pruning results
            print_results(pruning_results)
            
            # Save pruning results
            save_pruning_results(pruning_results)
            
            # 3. Compare Results
            print(f"\n{'='*60}")
            print(f"Performance Comparison (Dataset: {dataset_name})")
            print(f"{'='*60}")
            print(f"Accuracy Comparison:")
            print(f"  Original KNN-DTW: {original_accuracy:.6f} ({original_accuracy*100:.2f}%)")
            print(f"  Pruning KNN-DTW: {pruning_accuracy:.6f} ({pruning_accuracy*100:.2f}%)")
            print(f"  Accuracy Change: {pruning_accuracy - original_accuracy:+.4f} ({((pruning_accuracy/original_accuracy)-1)*100:+.2f}%)")
            
            print(f"\nTime Comparison:")
            print(f"  Original KNN-DTW: {original_time:.2f} seconds")
            print(f"  Pruning KNN-DTW: {pruning_time:.2f} seconds")
            print(f"  Speedup Factor: {speedup_factor:.2f}x")
            print(f"  Time Savings: {((original_time-pruning_time)/original_time)*100:.1f}%")
            
            if pruning_rate > 0:
                print(f"\nPruning Effectiveness:")
                print(f"  Pruning Rate: {pruning_rate:.6f} ({pruning_rate*100:.2f}%)")
                print(f"  Pruning Threshold: {PRUNING_COEFFICIENT * same_class_median:.6f}")
            
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\nExperiment Complete! Results saved to:")
    print("C:\\Users\\luoyu\\Desktop\\KNN_DTW_tuning_result\\pruning_results.csv")

if __name__ == "__main__":
    main()
