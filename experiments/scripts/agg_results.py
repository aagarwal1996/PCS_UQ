# Standard imports
import pandas as pd
import numpy as np
import argparse
import os
import pickle
import json
from pathlib import Path

# Experiment configs
from experiments.configs.regression_consts import VALID_UQ_METHODS, VALID_ESTIMATORS, MODELS, DATASETS, SINGLE_CONFORMAL_METHODS

def agg_results_dataset(dataset_name, results_dir="experiments/results/regression", train_size=0.8):
    """
    Aggregate results across all seeds for a specific dataset.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset to aggregate results for.
    results_dir : str, default="experiments/results/regression"
        Directory where results are stored.
    train_size : float, default=0.8
        Train size used in the experiments.
        
    Returns:
    --------
    dict
        Dictionary containing aggregated metrics for each method.
    """
    results_path = Path(results_dir)
    dataset_path = results_path / dataset_name
    
    if not dataset_path.exists():
        print(f"Dataset path {dataset_path} does not exist.")
        return {}
    
    # Find all unique methods by looking at the metrics files
    all_files = list(dataset_path.glob(f"*_seed_*_train_size_{train_size}_metrics.pkl"))
    methods = set()
    for file in all_files:
        # Extract method name from filename
        filename = file.name
        method_name = filename.split("_seed_")[0]
        methods.add(method_name)
    
    aggregated_results = {}
    
    # For each method, aggregate results across seeds
    for method in methods:
        print(f"Aggregating results for {method} on {dataset_name}")
        
        # Find all seed files for this method
        method_files = list(dataset_path.glob(f"{method}_seed_*_train_size_{train_size}_metrics.pkl"))
        
        if not method_files:
            print(f"No files found for method {method}. Skipping.")
            continue
        
        # Load all metrics
        all_metrics = []
        seeds = []
        for file in method_files:
            # Extract seed from filename
            filename = file.name
            seed = int(filename.split("_seed_")[1].split("_train_size_")[0])
            seeds.append(seed)
            
            with open(file, 'rb') as f:
                metrics = pickle.load(f)
                all_metrics.append(metrics)
        
        # Calculate mean and std for each metric
        agg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            agg_metrics[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values,
                'seeds': seeds
            }
        
        # Save aggregated metrics
        agg_file = f'{dataset_path}/{method}_train_size_{train_size}_agg_metrics.pkl'
        #with open(agg_file, 'wb') as f:
        #    pickle.dump(agg_metrics, f)
        
        print(f"Saved aggregated metrics to {agg_file}")
        
        # Add to results dictionary
        aggregated_results[method] = agg_metrics
    
    return aggregated_results

def agg_results(results_dir="experiments/results/regression", train_size=0.8):
    """
    Aggregate results across all seeds for all datasets.
    
    Parameters:
    -----------
    results_dir : str, default="experiments/results/regression"
        Directory where results are stored.
    train_size : float, default=0.8
        Train size used in the experiments.
        
    Returns:
    --------
    dict
        Dictionary containing aggregated metrics for each dataset and method.
    """
    results_path = Path(results_dir)
    
    # Get all datasets
    datasets = [d.name for d in results_path.iterdir() if d.is_dir()]
    
    all_results = {}
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        dataset_results = agg_results_dataset(dataset, results_dir, train_size)
        all_results[dataset] = dataset_results
    
    # Save overall results
    overall_results_file = f'{results_path}/all_datasets_train_size_{train_size}_agg_metrics.pkl'
    #with open(overall_results_file, 'wb') as f:
    #    pickle.dump(all_results, f)
    
    print(f"Saved overall aggregated metrics to {overall_results_file}")
    
    return all_results

def calculate_avg_width_reduction(all_results):
    """
    Calculate the average width reduction of pcs_uq compared to conformal methods across all datasets.
    
    Parameters:
    -----------
    all_results : dict
        Dictionary containing results for all datasets.
        
    Returns:
    --------
    dict
        Dictionary containing average width reduction for each conformal method.
    """
    # Track reductions for each conformal method
    method_reductions = {}
    method_counts = {}
    
    for dataset, dataset_results in all_results.items():
        # Check if pcs_uq exists for this dataset
        if 'pcs_uq' not in dataset_results or 'mean_width_scaled' not in dataset_results['pcs_uq']:
            continue
            
        pcs_width = dataset_results['pcs_uq']['mean_width_scaled']['mean']
        
        # Find all conformal methods for this dataset
        conformal_methods = [m for m in dataset_results.keys() if 'conformal' in m]
        
        for method in conformal_methods:
            if 'mean_width_scaled' not in dataset_results[method]:
                continue
                
            conf_width = dataset_results[method]['mean_width_scaled']['mean']
            pct_decrease = ((conf_width - pcs_width) / conf_width) * 100
            
            if method not in method_reductions:
                method_reductions[method] = 0
                method_counts[method] = 0
                
            method_reductions[method] += pct_decrease
            method_counts[method] += 1
    
    # Calculate averages
    avg_reductions = {}
    for method in method_reductions:
        if method_counts[method] > 0:
            avg_reductions[method] = method_reductions[method] / method_counts[method]
    
    return avg_reductions

def find_best_conformal_methods(all_results, num_best=2):
    """
    Find the best conformal methods across all datasets based on average mean_width_scaled.
    
    Parameters:
    -----------
    all_results : dict
        Dictionary containing results for all datasets.
    num_best : int, default=2
        Number of best methods to return.
        
    Returns:
    --------
    list
        List of best conformal method names.
    """
    # Collect all conformal methods across datasets
    all_conformal_methods = set()
    for dataset, dataset_results in all_results.items():
        conformal_methods = {k for k in dataset_results.keys() if 'conformal' in k}
        all_conformal_methods.update(conformal_methods)
    
    # Calculate average mean_width_scaled for each method across datasets
    method_avg_widths = {}
    for method in all_conformal_methods:
        widths = []
        for dataset, dataset_results in all_results.items():
            if method in dataset_results and 'mean_width_scaled' in dataset_results[method]:
                widths.append(dataset_results[method]['mean_width_scaled']['mean'])
        
        if widths:  # Only include methods that have width metrics in at least one dataset
            method_avg_widths[method] = sum(widths) / len(widths)
    
    # Sort methods by average width
    sorted_methods = sorted(method_avg_widths.items(), key=lambda x: x[1])
    
    # Return the best methods
    return [method for method, _ in sorted_methods[:num_best]]

def print_methods_comparison(all_results, methods_to_compare):
    """
    Print performance comparison of specified methods across all datasets.
    
    Parameters:
    -----------
    all_results : dict
        Dictionary containing results for all datasets.
    methods_to_compare : list
        List of method names to compare.
    """
    print("\n\n===== PERFORMANCE COMPARISON ACROSS DATASETS =====")
    
    for dataset, dataset_results in all_results.items():
        print(f"\n\n{dataset.upper()}:")
        
        # Extract width values for percentage comparison
        width_values = {}
        
        for method in methods_to_compare:
            if method in dataset_results:
                print(f"\n  {method}:")
                
                # Print coverage if available
                if 'coverage' in dataset_results[method]:
                    coverage = dataset_results[method]['coverage']
                    print(f"    coverage: {coverage['mean']:.3f} ± {coverage['std']:.3f}")
                
                # Print mean_width_scaled if available
                if 'mean_width_scaled' in dataset_results[method]:
                    width = dataset_results[method]['mean_width_scaled']
                    print(f"    mean_width_scaled: {width['mean']:.3f} ± {width['std']:.3f}")
                    width_values[method] = width['mean']
                
                # Print other key metrics based on method type
                if 'pcs' in method:
                    if 'rmse' in dataset_results[method]:
                        rmse = dataset_results[method]['rmse']
                        print(f"    rmse: {rmse['mean']:.3f} ± {rmse['std']:.3f}")
                    if 'r2' in dataset_results[method]:
                        r2 = dataset_results[method]['r2']
                        print(f"    r2: {r2['mean']:.3f} ± {r2['std']:.3f}")
            else:
                print(f"\n  {method}: Not available for this dataset")
        
        # Calculate percentage decrease in width for pcs_uq vs best conformal methods
        if 'pcs_uq' in width_values:
            pcs_width = width_values['pcs_uq']
            conformal_methods = [m for m in width_values.keys() if 'conformal' in m]
            
            if conformal_methods:
                print("\n  Width reduction by pcs_uq:")
                for method in conformal_methods:
                    if method in width_values:
                        conf_width = width_values[method]
                        pct_decrease = ((conf_width - pcs_width) / conf_width) * 100
                        print(f"    vs {method}: {pct_decrease:.3f}% reduction")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="Name of dataset to run experiments on. If None, aggregates all datasets.")
    parser.add_argument("--results_dir", type=str, default="experiments/results/regression", help="Directory where results are stored")
    parser.add_argument("--train_size", type=float, default=0.8, help="Train size used in the experiments")
    args = parser.parse_args()
    
    if args.dataset:
        results = agg_results_dataset(args.dataset, args.results_dir, args.train_size)
    else:
        results = agg_results(args.results_dir, args.train_size)
    
    # Print a summary of the results
    print("\nSummary of aggregated results:")
    if args.dataset:
        # Get best 2 conformal methods based on width
        conformal_methods = {k:v for k,v in results.items() if 'conformal' in k}
        
        # Use mean_width_scaled instead of width
        sorted_by_width = sorted(conformal_methods.items(), 
                                key=lambda x: x[1]['mean_width_scaled']['mean'] 
                                if 'mean_width_scaled' in x[1] else float('inf'))[:2]
        
        # Print best 2 conformal methods by width
        print("\nBest conformal methods by mean width (scaled):")
        for method, metrics in sorted_by_width:
            print(f"\n{method}:")
            print(f"  coverage: {metrics['coverage']['mean']:.4f} ± {metrics['coverage']['std']:.4f}")
            if 'mean_width_scaled' in metrics:
                print(f"  mean_width_scaled: {metrics['mean_width_scaled']['mean']:.4f} ± {metrics['mean_width_scaled']['std']:.4f}")
            else:
                print(f"  mean_width_scaled: not available")
                
        # Print PCS methods
        for method in ['pcs_uq', 'pcs_oob']:
            if method in results:
                print(f"\n{method}:")
                for metric, values in results[method].items():
                    print(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f}")
    else:
        for dataset, dataset_results in results.items():
            print(f"\n{dataset}:")
            
            # Get best 2 conformal methods based on width
            conformal_methods = {k:v for k,v in dataset_results.items() if 'conformal' in k}
            
            # Use mean_width_scaled instead of width
            sorted_by_width = sorted(conformal_methods.items(),
                                    key=lambda x: x[1]['mean_width_scaled']['mean'] 
                                    if 'mean_width_scaled' in x[1] else float('inf'))[:2]
            
            # Print best 2 conformal methods by width
            print(f"  Best conformal methods by mean width (scaled):")
            for method, metrics in sorted_by_width:
                print(f"    {method}:")
                print(f"      coverage: {metrics['coverage']['mean']:.4f} ± {metrics['coverage']['std']:.4f}")
                if 'mean_width_scaled' in metrics:
                    print(f"      mean_width_scaled: {metrics['mean_width_scaled']['mean']:.4f} ± {metrics['mean_width_scaled']['std']:.4f}")
                else:
                    print(f"      mean_width_scaled: not available")
                
            # Print PCS methods
            for method in ['pcs_uq', 'pcs_oob']:
                if method in dataset_results:
                    print(f"  {method}:")
                    for metric, values in dataset_results[method].items():
                        print(f"    {metric}: {values['mean']:.4f} ± {values['std']:.4f}")
    
    # Always find and compare best methods across datasets when running on all datasets
    if not args.dataset:
        best_conformal_methods = find_best_conformal_methods(results)
        methods_to_compare = best_conformal_methods + ['pcs_uq', 'pcs_oob']
        print(f"\nBest conformal methods across all datasets: {best_conformal_methods}")
        print_methods_comparison(results, methods_to_compare)
        
        # Calculate and print average width reduction across all datasets
        avg_reductions = calculate_avg_width_reduction(results)
        if avg_reductions:
            print("\n\n===== AVERAGE WIDTH REDUCTION ACROSS ALL DATASETS =====")
            for method, reduction in avg_reductions.items():
                print(f"pcs_uq vs {method}: {reduction:.3f}% reduction")
            
            # Calculate average reduction across all conformal methods
            if avg_reductions:
                overall_avg = sum(avg_reductions.values()) / len(avg_reductions)
                print(f"\nAverage reduction across all conformal methods: {overall_avg:.3f}%")
    else:
        # For single dataset, compare pcs_uq with best conformal methods
        conformal_methods = [k for k in results.keys() if 'conformal' in k]
        sorted_by_width = sorted(conformal_methods, 
                              key=lambda x: results[x]['mean_width_scaled']['mean'] 
                              if 'mean_width_scaled' in results[x] else float('inf'))[:2]
        methods_to_compare = sorted_by_width + ['pcs_uq', 'pcs_oob']
        print(f"\nComparing methods: {methods_to_compare}")
        print_methods_comparison({args.dataset: results}, methods_to_compare)
