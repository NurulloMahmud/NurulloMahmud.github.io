"""
Bookstore ETL Pipeline - Main Orchestrator

This script runs the complete ETL pipeline for all datasets
and generates the results and charts.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import BASE_DIR, DATA_FOLDERS, CHARTS_DIR, RESULTS_DIR
from src.extract import load_all_data
from src.transform import transform_all
from src.analyze import run_analysis
from src.visualize import plot_daily_revenue


def run_pipeline(data_folder: Path, dataset_name: str) -> dict:
    """
    Run the complete ETL pipeline for a single dataset.
    
    Args:
        data_folder: Path to the data folder (e.g., /project/data/DATA1)
        dataset_name: Name of the dataset (DATA1, DATA2, DATA3)
    
    Returns:
        Dictionary with all analysis results
    """
    print(f"\n{'='*60}")
    print(f"  Processing {dataset_name}")
    print('='*60)
    
    # EXTRACT: Load raw data from files
    print("\n  [1/4] EXTRACT - Loading data from files...")
    
    users_raw, orders_raw, books_raw = load_all_data(str(data_folder))
    
    print(f"Loaded {len(users_raw):,} users")
    print(f"Loaded {len(orders_raw):,} orders")
    print(f"Loaded {len(books_raw):,} books")
    
    # TRANSFORM: Clean and prepare data
    print("\n  [2/4] TRANSFORM - Cleaning and preparing data...")
    
    users, orders, books = transform_all(users_raw, orders_raw, books_raw)
    
    print(f"Transformed users: {len(users):,} records")
    print(f"Transformed orders: {len(orders):,} valid records")
    print(f"Transformed books: {len(books):,} records")
    
    # ANALYZE: Run business analysis
    print("\n  [3/4] ANALYZE - Running business analysis...")
    
    results = run_analysis(users, orders, books)
    results['dataset'] = dataset_name
    
    # print summary
    print(f"Top revenue day: {results['top_5_revenue_days'][0]['date']} (${results['top_5_revenue_days'][0]['revenue']:,.2f})")
    print(f"Unique users: {results['unique_users_count']:,}")
    print(f"Unique author sets: {results['unique_author_sets_count']:,}")
    print(f"Most popular author: {results['most_popular_author']}")
    print(f"Top customer IDs: {results['top_customer_ids']}")
    
    # VISUALIZE: Create charts
    print("\n  [4/4] VISUALIZE - Creating charts...")
    
    # ensure output directory exists
    charts_dir = PROJECT_ROOT / CHARTS_DIR
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    # create daily revenue chart
    chart_path = plot_daily_revenue(
        results['daily_revenue'],
        str(charts_dir),
        dataset_name
    )
    
    print(f"Saved chart: {chart_path}")
    
    # add chart path to results (relative path for dashboard)
    results['chart_path'] = f"charts/{dataset_name.lower()}_daily_revenue.png"
    
    return results


def save_results(results: dict, output_dir: Path, dataset_name: str) -> str:
    """
    Save analysis results to a JSON file.
    """
    # ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # create filename
    filename = f"{dataset_name.lower()}_results.json"
    filepath = output_dir / filename
    
    # save as JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return str(filepath)


def main():
    """
    Run the complete pipeline for all datasets.
    """
    print("\n" + "="*60)
    print("  BOOKSTORE ETL PIPELINE")
    print("="*60)
    print(f"\n  Project root: {PROJECT_ROOT}")
    print(f"  Data folders: {DATA_FOLDERS}")
    
    all_results = {}
    
    # process each dataset
    for dataset_name in DATA_FOLDERS:
        # build path to data folder
        data_folder = BASE_DIR / 'data' / dataset_name
        
        # check if folder exists
        if not data_folder.exists():
            print(f"\n     WARNING: {data_folder} not found, skipping...")
            continue
        
        # run the pipeline
        results = run_pipeline(data_folder, dataset_name)
        
        # save results to JSON
        results_dir = PROJECT_ROOT / RESULTS_DIR
        saved_path = save_results(results, results_dir, dataset_name)
        print(f"\n Saved results: {saved_path}")
        
        # store for summary
        all_results[dataset_name] = results
    
    # FINAL SUMMARY
    print("\n" + "="*60)
    print("  PIPELINE COMPLETE - SUMMARY")
    print("="*60)
    
    for dataset_name, results in all_results.items():
        print(f"\n  {dataset_name}:")
        print(f" Top 5 Revenue Days:")
        for i, day in enumerate(results['top_5_revenue_days'], 1):
            print(f" {i}. {day['date']} - ${day['revenue']:,.2f}")
        print(f"    Unique Users: {results['unique_users_count']:,}")
        print(f"    Unique Author Sets: {results['unique_author_sets_count']:,}")
        print(f"    Most Popular Author: {results['most_popular_author']} ({results['most_popular_author_books_sold']} books)")
        print(f"    Top Customer: {results['top_customer_ids']} (${results['top_customer_total_spent']:,.2f})")
    
    print("\n" + "="*60)
    print("  OUTPUT FILES")
    print("="*60)
    print(f"\n  Results JSON files: {PROJECT_ROOT / RESULTS_DIR}")
    print(f"  Chart PNG files:    {PROJECT_ROOT / CHARTS_DIR}")
    
    print("\n" + "="*60)
    print("All done!")
    print("="*60 + "\n")
    
    return all_results


if __name__ == '__main__':
    main()