#!/usr/bin/env python3
"""
Extract full loss time series per model from MLflow experiment.

Usage:
    python extract_mlflow_losses.py --experiment_name "paper-2025-07-22_training"
"""

import click
import mlflow
import pandas as pd
import numpy as np
import os
import shutil
from mlflow.tracking import MlflowClient


@click.command()
@click.option("--experiment_name", required=True, help="MLflow experiment name")
@click.option("--output_file", default="mlflow_losses.csv", help="Output CSV file for summary")
@click.option("--output_timeseries", default="mlflow_loss_timeseries.csv", help="Output CSV file for full time series")
@click.option("--samples_dir", default="../influpaint_res/all_npy/", help="Directory to save extracted sample artifacts")
@click.option("--download_samples", is_flag=True, help="Download and save sample artifacts from MLflow runs")
def main(experiment_name, output_file, output_timeseries, samples_dir, download_samples):
    """Extract loss time series from MLflow experiment"""
    
    # Initialize MLflow client
    client = MlflowClient()
    
    # Get experiment by name
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found")
            return
    except Exception as e:
        print(f"Error finding experiment: {e}")
        return
    
    print(f"Found experiment: {experiment_name}")
    print(f"Experiment ID: {experiment.experiment_id}")
    
    # Get all runs from the experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=["start_time DESC"]
    )
    
    print(f"Found {len(runs)} runs in experiment")
    
    # Extract summary data and full time series
    summary_data = []
    timeseries_data = []
    
    for run in runs:
        run_id = run.info.run_id
        scenario_id = run.data.params.get('scenario_id')
        scenario_string = run.data.params.get('scenario_string')
        
        # Summary data
        run_summary = {
            'run_id': run_id,
            'run_name': run.data.tags.get('mlflow.runName', ''),
            'scenario_id': scenario_id,
            'scenario_string': scenario_string,
            'final_loss': run.data.metrics.get('final_loss'),
            'avg_loss_last_100': run.data.metrics.get('avg_loss_last_100'),
            'ddpm_name': run.data.params.get('ddpm_name'),
            'dataset_name': run.data.params.get('dataset_name'),
            'transform_name': run.data.params.get('transform_name'),
            'enrich_name': run.data.params.get('enrich_name'),
            'epochs': run.data.params.get('epochs'),
            'batch_size': run.data.params.get('batch_size'),
            'dataset_size': run.data.params.get('dataset_size'),
            'training_completed': run.data.metrics.get('training_completed'),
            'start_time': run.info.start_time,
            'end_time': run.info.end_time,
            'status': run.info.status
        }
        summary_data.append(run_summary)
        
        # Get full loss time series for this run

            # Note that there are also step loss
        loss_history = client.get_metric_history(run_id, "epoch_loss")
        print(f"Found {len(loss_history)} loss steps for run {scenario_id}")
        for metric in loss_history:
            timeseries_data.append({
                'run_id': run_id,
                'scenario_id': scenario_id,
                'scenario_string': scenario_string,
                'step': metric.step,
                'timestamp': metric.timestamp,
                'loss': metric.value
            })
    
    # Create DataFrames
    summary_df = pd.DataFrame(summary_data)
    timeseries_df = pd.DataFrame(timeseries_data)
    
    # Filter completed runs
    completed_summary = summary_df[
        (summary_df['training_completed'] == 1.0) & 
        (summary_df['final_loss'].notna())
    ].copy()
    
    print(f"\nCompleted runs with final_loss: {len(completed_summary)}")
    
    if len(completed_summary) > 0:
        # Sort by scenario_id
        completed_summary = completed_summary.sort_values('scenario_id')
        completed_summary.to_csv(output_file, index=False)
        print(f"Summary saved to: {output_file}")
    
    # Save time series data
    if len(timeseries_df) > 0:
        timeseries_df = timeseries_df.sort_values(['scenario_id', 'step'])
        timeseries_df.to_csv(output_timeseries, index=False)
        print(f"\nTime series data saved to: {output_timeseries}")
        print(f"Total time series points: {len(timeseries_df)}")
        print(f"Runs with time series: {timeseries_df['run_id'].nunique()}")
        
        # Show time series summary
        if len(timeseries_df) > 0:
            print(f"Steps per run range: {timeseries_df.groupby('run_id')['step'].count().min()} - {timeseries_df.groupby('run_id')['step'].count().max()}")
            print("\nTime series sample:")
            print(timeseries_df[['scenario_id', 'step', 'loss']].head(10).to_string(index=False))
    else:
        print("\nNo time series data found. Loss may be logged under different metric names or not logged step-by-step.")
    
    # Download sample artifacts if requested
    if download_samples:
        print(f"\nDownloading sample artifacts to {samples_dir}...")
        download_sample_artifacts(client, runs, samples_dir)
    
    return completed_summary, timeseries_df


def download_sample_artifacts(client, runs, samples_dir):
    """Download raw_samples and inverse_transformed_samples from MLflow runs"""
    
    # Create output directory
    os.makedirs(samples_dir, exist_ok=True)
    
    downloaded_count = 0
    failed_count = 0
    
    for run in runs:
        run_id = run.info.run_id
        scenario_id = run.data.params.get('scenario_id', 'unknown')
        scenario_string = run.data.params.get('scenario_string', f'scenario_{scenario_id}')
        
        try:
            # List artifacts for this run
            artifacts = client.list_artifacts(run_id, path="samples")
            
            if not artifacts:
                print(f"  No sample artifacts found for {scenario_string} (run {run_id})")
                continue
                
            print(f"  Processing {scenario_string} (run {run_id})...")
            
            for artifact in artifacts:
                if artifact.path.endswith('.npy'):
                    artifact_name = os.path.basename(artifact.path)
                    
                    # Create scenario-specific filename using full scenario string
                    if 'raw_samples' in artifact_name:
                        output_filename = f"raw_samples_{scenario_string}.npy"
                    elif 'inverse_transformed_samples' in artifact_name:
                        output_filename = f"inverse_transformed_samples_{scenario_string}.npy"
                    else:
                        continue  # Skip other .npy files
                    
                    output_path = os.path.join(samples_dir, output_filename)
                    
                    # Skip if file already exists
                    if os.path.exists(output_path):
                        print(f"    Skipped (exists): {output_filename}")
                        continue
                    
                    print(f"    Downloading {artifact_name}...")
                    
                    try:
                        # Download artifact to temporary location first
                        temp_artifact_path = client.download_artifacts(run_id, artifact.path)
                        
                        # Copy to final location
                        shutil.copy2(temp_artifact_path, output_path)
                        print(f"    Downloaded: {output_filename}")
                        downloaded_count += 1
                    except Exception as download_error:
                        print(f"    Failed to download {artifact_name}: {download_error}")
                        continue
                    
        except Exception as e:
            print(f"  Failed to download artifacts for {scenario_string}: {e}")
            failed_count += 1
            continue
    
    print(f"\nSample artifact download complete:")
    print(f"  Successfully downloaded: {downloaded_count} files")
    print(f"  Failed downloads: {failed_count} runs")
    print(f"  Saved to: {samples_dir}")


if __name__ == '__main__':
    main()