import numpy as np
from Functions import txt_loader, cdtw
import time
from collections import defaultdict
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def analyze_similarity_thresholds(dataset_name, w):
    """
    Analyze CDTW distances between same-class and different-class time series using LOOCV (Leave-One-Out Cross-Validation).
    
    Args:
        dataset_name: Name of the dataset
        w: Warping window parameter for CDTW
    
    Returns:
        dict: Dictionary containing analysis results
    """
    print(f"Start analyzing dataset: {dataset_name}")
    
    # Load training data
    train_data = txt_loader(f'C:\\Users\\luoyu\\Desktop\\Univariate_arff\\{dataset_name}\\{dataset_name}_TRAIN.txt')
    
    # Organize data by class
    class_data = defaultdict(list)
    for i, template in enumerate(train_data):
        label = int(template[0])  # First column is the label
        time_series = template[1:]  # Remaining columns are time series data
        class_data[label].append((i, time_series))
    
    print(f"Dataset {dataset_name} contains {len(class_data)} classes")
    for label, templates in class_data.items():
        print(f"  Class {label}: {len(templates)} templates")
    
    # Store distance statistics
    same_class_distances = []
    different_class_distances = []
    
    # LOOCV: Leave one sample out as test, others as templates
    total_samples = len(train_data)
    processed = 0
    
    for test_idx in range(total_samples):
        test_template = train_data[test_idx]
        test_label = int(test_template[0])
        test_series = np.array(test_template[1:])
        
        # Calculate CDTW distance to all other templates
        for template_idx in range(total_samples):
            if template_idx == test_idx:  # Skip self
                continue
                
            template_template = train_data[template_idx]
            template_label = int(template_template[0])
            template_series = np.array(template_template[1:])
            
            # Calculate CDTW distance
            try:
                distance = cdtw(template_series, test_series, w)
                
                # Classify distance based on class relationship
                if template_label == test_label:
                    same_class_distances.append(distance)
                else:
                    different_class_distances.append(distance)
                    
            except Exception as e:
                print(f"Error calculating CDTW distance: {e}")
                continue
        
        processed += 1
        if processed % 10 == 0:
            print(f"Processed {processed}/{total_samples} templates")
    
    # Calculate statistics
    results = {
        'dataset_name': dataset_name,
        'w': w,
        'same_class': {
            'count': len(same_class_distances),
            'mean': np.mean(same_class_distances) if same_class_distances else 0,
            'std': np.std(same_class_distances) if same_class_distances else 0,
            'min': np.min(same_class_distances) if same_class_distances else 0,
            'max': np.max(same_class_distances) if same_class_distances else 0,
            'median': np.median(same_class_distances) if same_class_distances else 0
        },
        'different_class': {
            'count': len(different_class_distances),
            'mean': np.mean(different_class_distances) if different_class_distances else 0,
            'std': np.std(different_class_distances) if different_class_distances else 0,
            'min': np.min(different_class_distances) if different_class_distances else 0,
            'max': np.max(different_class_distances) if different_class_distances else 0,
            'median': np.median(different_class_distances) if different_class_distances else 0
        }
    }
    
    # Calculate distance ratio
    if results['different_class']['mean'] > 0:
        results['distance_ratio'] = results['same_class']['mean'] / results['different_class']['mean']
    else:
        results['distance_ratio'] = float('inf')
    
    # Add raw distance data for visualization
    results['same_class_distances'] = same_class_distances
    results['different_class_distances'] = different_class_distances
    
    return results

def print_results(results):
    """Print analysis results"""
    print("\n" + "="*60)
    print(f"Dataset: {results['dataset_name']}")
    print(f"Warping Window (w): {results['w']}")
    print("="*60)
    
    print("\nSame-Class Time Series CDTW Distance Statistics:")
    print(f"  Count: {results['same_class']['count']}")
    print(f"  Mean Distance: {results['same_class']['mean']:.6f}")
    print(f"  Std Dev: {results['same_class']['std']:.6f}")
    print(f"  Min Distance: {results['same_class']['min']:.6f}")
    print(f"  Max Distance: {results['same_class']['max']:.6f}")
    print(f"  Median: {results['same_class']['median']:.6f}")
    
    print("\nDifferent-Class Time Series CDTW Distance Statistics:")
    print(f"  Count: {results['different_class']['count']}")
    print(f"  Mean Distance: {results['different_class']['mean']:.6f}")
    print(f"  Std Dev: {results['different_class']['std']:.6f}")
    print(f"  Min Distance: {results['different_class']['min']:.6f}")
    print(f"  Max Distance: {results['different_class']['max']:.6f}")
    print(f"  Median: {results['different_class']['median']:.6f}")
    
    print(f"\nDistance Ratio (Same-Class Mean / Diff-Class Mean): {results['distance_ratio']:.6f}")
    
    if results['distance_ratio'] < 1:
        print("✓ Same-class distance is less than different-class distance, CDTW effectively distinguishes classes")
    else:
        print("✗ Same-class distance is greater than different-class distance, CDTW distinction is poor")



def save_results_to_csv(results, csv_filename="similarity_threshold_results.csv"):
    """Save results to CSV file"""
    # Ensure directory exists
    csv_dir = "C:\\Users\\luoyu\\Desktop\\KNN_DTW_tuning_result"
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    
    csv_path = os.path.join(csv_dir, csv_filename)
    
    # Check if file exists to determine whether to write header
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write header if file doesn't exist
        if not file_exists:
            writer.writerow([
                'Dataset', 'W', 'Same_Class_Count', 'Same_Class_Mean', 'Same_Class_Std', 
                'Same_Class_Min', 'Same_Class_Max', 'Same_Class_Median',
                'Different_Class_Count', 'Different_Class_Mean', 'Different_Class_Std',
                'Different_Class_Min', 'Different_Class_Max', 'Different_Class_Median',
                'Distance_Ratio', 'Effectiveness'
            ])
        
        # Write data row
        effectiveness = "Effective" if results['distance_ratio'] < 1 else "Ineffective"
        writer.writerow([
            results['dataset_name'],
            results['w'],
            results['same_class']['count'],
            f"{results['same_class']['mean']:.6f}",
            f"{results['same_class']['std']:.6f}",
            f"{results['same_class']['min']:.6f}",
            f"{results['same_class']['max']:.6f}",
            f"{results['same_class']['median']:.6f}",
            results['different_class']['count'],
            f"{results['different_class']['mean']:.6f}",
            f"{results['different_class']['std']:.6f}",
            f"{results['different_class']['min']:.6f}",
            f"{results['different_class']['max']:.6f}",
            f"{results['different_class']['median']:.6f}",
            f"{results['distance_ratio']:.6f}",
            effectiveness
        ])
    
    print(f"Results saved to CSV file: {csv_path}")

def visualize_distance_distribution(results, save_plot=True):
    """
    Visualize distribution of same-class and different-class distances
    
    Args:
        results: Analysis results dictionary
        save_plot: Whether to save the plot
    """
    # Font settings (SimHei kept for compatibility if needed, but text is now English)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    dataset_name = results['dataset_name']
    same_distances = results['same_class_distances']
    different_distances = results['different_class_distances']
    
    # Set colors
    same_color = '#2E86AB'  # Blue
    diff_color = '#A23B72'  # Red
    
    # 1. Histogram
    ax1.hist(same_distances, bins=30, alpha=0.7, color=same_color, 
             label=f'Same Class (n={len(same_distances)})', density=True)
    ax1.hist(different_distances, bins=30, alpha=0.7, color=diff_color, 
             label=f'Different Class (n={len(different_distances)})', density=True)
    
    ax1.set_xlabel('CDTW Distance')
    ax1.set_ylabel('Density')
    ax1.set_title(f'{dataset_name} - CDTW Distance Distribution Histogram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Boxplot
    data_to_plot = [same_distances, different_distances]
    labels = ['Same Class', 'Different Class']
    colors = [same_color, diff_color]
    
    bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('CDTW Distance')
    ax2.set_title(f'{dataset_name} - CDTW Distance Boxplot')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    same_mean = np.mean(same_distances)
    diff_mean = np.mean(different_distances)
    
    # Add mean lines to boxplot
    ax2.axhline(y=same_mean, color=same_color, linestyle='--', alpha=0.8, 
                label=f'Same Class Mean: {same_mean:.2f}')
    ax2.axhline(y=diff_mean, color=diff_color, linestyle='--', alpha=0.8, 
                label=f'Different Class Mean: {diff_mean:.2f}')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    if save_plot:
        # Ensure directory exists
        plot_dir = "C:\\Users\\luoyu\\Desktop\\KNN_DTW_tuning_result"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        plot_path = os.path.join(plot_dir, f"{dataset_name}_distance_distribution.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Distance distribution plot saved to: {plot_path}")
    
    plt.show()

def create_detailed_visualization(results, save_plot=True):
    """
    Create detailed visualization including distance interval statistics
    
    Args:
        results: Analysis results dictionary
        save_plot: Whether to save the plot
    """
    # Font settings
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    dataset_name = results['dataset_name']
    same_distances = results['same_class_distances']
    different_distances = results['different_class_distances']
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Distance Interval Statistics
    ax1 = plt.subplot(2, 3, 1)
    
    # Calculate distance intervals
    all_distances = same_distances + different_distances
    min_dist = min(all_distances)
    max_dist = max(all_distances)
    
    # Create 10 equal-width bins
    bins = np.linspace(min_dist, max_dist, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Count frequency in each bin
    same_counts, _ = np.histogram(same_distances, bins=bins)
    diff_counts, _ = np.histogram(different_distances, bins=bins)
    
    x = np.arange(len(bin_centers))
    width = 0.35
    
    ax1.bar(x - width/2, same_counts, width, label='Same Class', alpha=0.7, color='#2E86AB')
    ax1.bar(x + width/2, diff_counts, width, label='Different Class', alpha=0.7, color='#A23B72')
    
    ax1.set_xlabel('Distance Interval')
    ax1.set_ylabel('Template Count')
    ax1.set_title(f'{dataset_name} - Distance Interval Distribution')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{bin_centers[i]:.1f}' for i in range(len(bin_centers))], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative Distribution Function (CDF)
    ax2 = plt.subplot(2, 3, 2)
    
    # Sort distances
    same_sorted = np.sort(same_distances)
    diff_sorted = np.sort(different_distances)
    
    # Calculate CDF
    same_cdf = np.arange(1, len(same_sorted) + 1) / len(same_sorted)
    diff_cdf = np.arange(1, len(diff_sorted) + 1) / len(diff_sorted)
    
    ax2.plot(same_sorted, same_cdf, label='Same Class', color='#2E86AB', linewidth=2)
    ax2.plot(diff_sorted, diff_cdf, label='Different Class', color='#A23B72', linewidth=2)
    
    ax2.set_xlabel('CDTW Distance')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title(f'{dataset_name} - Cumulative Distribution Function')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Distance Ratio Distribution
    ax3 = plt.subplot(2, 3, 3)
    
    # Calculate ratio for each distance relative to overall mean
    overall_mean = np.mean(all_distances)
    same_ratios = np.array(same_distances) / overall_mean
    diff_ratios = np.array(different_distances) / overall_mean
    
    ax3.hist(same_ratios, bins=30, alpha=0.7, color='#2E86AB', 
             label=f'Same Class (Mean={np.mean(same_ratios):.2f})', density=True)
    ax3.hist(diff_ratios, bins=30, alpha=0.7, color='#A23B72', 
             label=f'Different Class (Mean={np.mean(diff_ratios):.2f})', density=True)
    
    ax3.set_xlabel('Distance Ratio (relative to overall mean)')
    ax3.set_ylabel('Density')
    ax3.set_title(f'{dataset_name} - Distance Ratio Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistical Summary
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')
    
    # Create statistics summary text
    stats_text = f"""
    Dataset: {dataset_name}
    
    Same-Class Statistics:
    • Count: {len(same_distances)}
    • Mean: {np.mean(same_distances):.2f}
    • Std Dev: {np.std(same_distances):.2f}
    • Min: {np.min(same_distances):.2f}
    • Max: {np.max(same_distances):.2f}
    • Median: {np.median(same_distances):.2f}
    
    Different-Class Statistics:
    • Count: {len(different_distances)}
    • Mean: {np.mean(different_distances):.2f}
    • Std Dev: {np.std(different_distances):.2f}
    • Min: {np.min(different_distances):.2f}
    • Max: {np.max(different_distances):.2f}
    • Median: {np.median(different_distances):.2f}
    
    Distance Ratio: {results['distance_ratio']:.4f}
    Effectiveness: {'Effective' if results['distance_ratio'] < 1 else 'Ineffective'}
    """
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 5. Percentile Range Analysis
    ax5 = plt.subplot(2, 3, 5)
    
    # Calculate counts in percentile ranges
    ranges = [
        (0, np.percentile(all_distances, 25), '0-25%'),
        (np.percentile(all_distances, 25), np.percentile(all_distances, 50), '25-50%'),
        (np.percentile(all_distances, 50), np.percentile(all_distances, 75), '50-75%'),
        (np.percentile(all_distances, 75), np.percentile(all_distances, 100), '75-100%')
    ]
    
    same_range_counts = []
    diff_range_counts = []
    range_labels = []
    
    for low, high, label in ranges:
        same_count = sum(1 for d in same_distances if low <= d < high)
        diff_count = sum(1 for d in different_distances if low <= d < high)
        same_range_counts.append(same_count)
        diff_range_counts.append(diff_count)
        range_labels.append(label)
    
    x = np.arange(len(range_labels))
    width = 0.35
    
    ax5.bar(x - width/2, same_range_counts, width, label='Same Class', alpha=0.7, color='#2E86AB')
    ax5.bar(x + width/2, diff_range_counts, width, label='Different Class', alpha=0.7, color='#A23B72')
    
    ax5.set_xlabel('Distance Percentile Range')
    ax5.set_ylabel('Template Count')
    ax5.set_title(f'{dataset_name} - Percentile Distribution')
    ax5.set_xticks(x)
    ax5.set_xticklabels(range_labels)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Overlap Analysis
    ax6 = plt.subplot(2, 3, 6)
    
    # Calculate overlap
    same_mean = np.mean(same_distances)
    diff_mean = np.mean(different_distances)
    
    # Count how many same-class distances > diff-class mean, and vice versa
    same_above_diff_mean = sum(1 for d in same_distances if d > diff_mean)
    diff_below_same_mean = sum(1 for d in different_distances if d < same_mean)
    
    overlap_data = [
        same_above_diff_mean,
        diff_below_same_mean,
        len(same_distances) - same_above_diff_mean,
        len(different_distances) - diff_below_same_mean
    ]
    
    overlap_labels = [
        f'Same > Diff Mean\n({same_above_diff_mean})',
        f'Diff < Same Mean\n({diff_below_same_mean})',
        f'Same ≤ Diff Mean\n({len(same_distances) - same_above_diff_mean})',
        f'Diff ≥ Same Mean\n({len(different_distances) - diff_below_same_mean})'
    ]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    ax6.pie(overlap_data, labels=overlap_labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax6.set_title(f'{dataset_name} - Distance Overlap Analysis')
    
    plt.tight_layout()
    
    # Save plot
    if save_plot:
        plot_dir = "C:\\Users\\luoyu\\Desktop\\KNN_DTW_tuning_result"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        plot_path = os.path.join(plot_dir, f"{dataset_name}_detailed_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Detailed analysis plot saved to: {plot_path}")
    
    plt.show()

def main():
    """Main function - can analyze multiple datasets"""
    
    # Dataset Configuration (Dataset Name, w parameter)
    datasets = [
        ("Car", 6)]

    #     ("CBF", 15),
    #     ("CricketX", 30),
    #     ("CricketY", 51),
    #     ("CricketZ", 15),
    #     ("ECG5000", 2),
    #     ("FaceAll", 4),
    #     ("FaceFour", 7),
    #     ("FacesUCR", 16),
    #     ("Fish", 17),
    #     ("FordA", 5),
    #     ("FordB", 5),
    #     ("Lightning2", 39),
    #     ("Lightning7", 3),
    #     ("MedicalImages", 16),
    #     ("Plane", 8),
    #     ("Trace", 1),
    #     ("TwoLeadECG", 2),
    #     ("Wafer", 58),
    #     ("Yoga", 30)
    # ]
    
    print("Starting CDTW Similarity Threshold Analysis...")
    print("Calculating CDTW distances between same-class and different-class time series using LOOCV")
    
    for dataset_name, w in datasets:
        try:
            # Check if data file exists
            data_path = f'C:\\Users\\luoyu\\Desktop\\Univariate_arff\\{dataset_name}\\{dataset_name}_TRAIN.txt'
            if not os.path.exists(data_path):
                print(f"Warning: Training file for dataset {dataset_name} does not exist, skipping")
                continue
            
            # Analyze dataset
            results = analyze_similarity_thresholds(dataset_name, w)
            
            # Print results
            print_results(results)
            
            # Save results to CSV
            save_results_to_csv(results)
            
            # Create visualizations
            print("\nGenerating visualization charts...")
            visualize_distance_distribution(results, save_plot=True)
            create_detailed_visualization(results, save_plot=True)
            
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            continue
    
    print("\nAnalysis Complete! Results saved to CSV file:")
    print("C:\\Users\\luoyu\\Desktop\\KNN_DTW_tuning_result\\similarity_threshold_results.csv")

if __name__ == "__main__":
    main()
