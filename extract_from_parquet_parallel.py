#!/usr/bin/env python3
import pandas as pd
import json
import numpy as np
from pathlib import Path
from PIL import Image
import io
import glob
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
import concurrent.futures
from functools import partial
import time
import shutil
import psutil
import gc
import os

def extract_episode_number(filename):
    """Extract episode number from filename like 'episode_000017.parquet'"""
    return filename.split('_')[1].split('.')[0]

def check_system_resources():
    """Check if system has enough resources to continue"""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Warn if memory usage is high
    if memory.percent > 95:
        print(f"WARNING: High memory usage ({memory.percent:.1f}%)")
        return False
    
    # Warn if disk space is low
    if disk.percent > 90:
        print(f"WARNING: Low disk space ({disk.percent:.1f}% used)")
        return False
    
    return True

def safe_delete_file(filepath, verify_replacement_exists=None):
    """
    Safely delete a file only after verifying the replacement exists and is valid.
    
    Args:
        filepath: Path to file to delete
        verify_replacement_exists: Path or directory to verify exists before deletion
    """
    try:
        if verify_replacement_exists:
            if isinstance(verify_replacement_exists, (str, Path)):
                replacement_path = Path(verify_replacement_exists)
                if not replacement_path.exists():
                    print(f"  ! Skipping deletion of {filepath.name} - replacement not found")
                    return False
                
                # If it's a directory, check it has content
                if replacement_path.is_dir():
                    contents = list(replacement_path.iterdir())
                    if len(contents) == 0:
                        print(f"  ! Skipping deletion of {filepath.name} - replacement directory empty")
                        return False
        
        # Perform the deletion
        filepath.unlink()
        return True
        
    except Exception as e:
        print(f"  ! Error deleting {filepath.name}: {str(e)}")
        return False

def process_episode_safe(parquet_path, output_dir, delete_after=True, show_progress=True):
    """
    Safe processing of a single episode parquet file with proper error handling.
    
    Args:
        parquet_path: Path to the parquet file
        output_dir: Base output directory
        delete_after: Whether to delete the parquet file after processing
        show_progress: Whether to show detailed progress for this episode
    """
    parquet_path = Path(parquet_path)
    episode_num = extract_episode_number(parquet_path.name)
    
    # Create episode directory
    episode_dir = Path(output_dir) / f"episode_{episode_num}"
    temp_episode_dir = Path(output_dir) / f"episode_{episode_num}_temp"
    
    # Use temporary directory first
    temp_episode_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Check system resources before starting
        if not check_system_resources():
            print(f"  ! Skipping episode {episode_num} due to resource constraints")
            return False, 0, 0
        
        # Load the parquet file
        print(f"\nLoading episode {episode_num}...")
        df = pd.read_parquet(parquet_path)
        total_frames = len(df)
        
        print(f"Processing episode {episode_num} ({total_frames} frames)...")
        
        # Image columns to extract
        image_columns = [
            'observation.images.cam_high',
            'observation.images.cam_low', 
            'observation.images.cam_left_wrist',
            'observation.images.cam_right_wrist'
        ]
        
        # Process in smaller chunks to manage memory
        chunk_size = min(25, total_frames)  # Process 50 frames at a time
        images_saved = 0
        metadata_saved = 0
        
        # Create progress bar for frame processing
        frame_pbar = None
        if show_progress:
            frame_pbar = tqdm(total=total_frames, desc=f"  Processing frames", 
                            unit="frame", leave=False, position=1)
        
        # Process frames in chunks
        for chunk_start in range(0, total_frames, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_frames)
            chunk_df = df.iloc[chunk_start:chunk_end]
            
            # Check resources periodically
            if chunk_start % (chunk_size * 4) == 0:  # Every 4 chunks
                if not check_system_resources():
                    print(f"  ! Stopping episode {episode_num} due to resource constraints")
                    break
            
            # Process chunk
            for idx, (frame_idx, row) in enumerate(chunk_df.iterrows()):
                actual_frame_idx = chunk_start + idx
                
                try:
                    # Save images for this frame
                    for col in image_columns:
                        if col in row and row[col] is not None:
                            cam_name = col.split('.')[-1]
                            img_data = row[col]
                            
                            filename = f"episode_{episode_num}_{actual_frame_idx:06d}_{cam_name}.png"
                            filepath = temp_episode_dir / filename
                            
                            if isinstance(img_data, dict) and 'bytes' in img_data:
                                img_bytes = img_data['bytes']
                                img = Image.open(io.BytesIO(img_bytes))
                                img.save(filepath, 'PNG', optimize=True)
                                img.close()  # Explicitly close to free memory
                                images_saved += 1
                            elif isinstance(img_data, Image.Image):
                                img_data.save(filepath, 'PNG', optimize=True)
                                images_saved += 1
                    
                    # Save metadata for this frame
                    metadata = {}
                    for col, value in row.items():
                        if not col.startswith('observation.images.'):
                            # Convert to JSON-serializable format
                            if isinstance(value, np.ndarray):
                                metadata[col] = value.tolist()
                            elif hasattr(value, 'tolist'):
                                metadata[col] = value.tolist()
                            elif isinstance(value, (list, tuple)):
                                metadata[col] = list(value)
                            else:
                                metadata[col] = value
                    
                    json_filename = f"episode_{episode_num}_{actual_frame_idx:06d}.json"
                    json_filepath = temp_episode_dir / json_filename
                    
                    with open(json_filepath, 'w') as f:
                        json.dump(metadata, f, separators=(',', ':'))
                    metadata_saved += 1
                    
                except Exception as e:
                    print(f"    Error processing frame {actual_frame_idx}: {str(e)}")
                    continue
                
                # Update progress
                if frame_pbar:
                    frame_pbar.update(1)
            
            # Force garbage collection after each chunk
            gc.collect()
        
        if frame_pbar:
            frame_pbar.close()
        
        # Clean up dataframe to free memory
        del df
        gc.collect()
        
        # Verify we have output files before proceeding
        temp_files = list(temp_episode_dir.iterdir())
        if len(temp_files) == 0:
            print(f"  ! No files created for episode {episode_num}")
            temp_episode_dir.rmdir()
            return False, 0, 0
        
        # Move from temp to final location atomically
        if episode_dir.exists():
            shutil.rmtree(episode_dir)
        temp_episode_dir.rename(episode_dir)
        
        print(f"  ✓ Saved {images_saved} PNG images")
        print(f"  ✓ Saved {metadata_saved} metadata entries")
        
        # Only delete original parquet file AFTER successful processing and verification
        if delete_after:
            # Double-check that our output directory has content
            output_files = list(episode_dir.iterdir())
            if len(output_files) > 0:
                if safe_delete_file(parquet_path, episode_dir):
                    print(f"  ✓ Deleted {parquet_path.name}")
            else:
                print(f"  ! Keeping {parquet_path.name} - no output files verified")
        
        return True, total_frames, images_saved
        
    except Exception as e:
        error_msg = f"Error processing {parquet_path.name}: {str(e)}"
        print(f"  ✗ {error_msg}")
        
        # Clean up temp directory on error
        if temp_episode_dir.exists():
            try:
                shutil.rmtree(temp_episode_dir)
            except:
                pass
        
        return False, 0, 0

def process_episode_wrapper(args):
    """Wrapper for multiprocessing with better error handling"""
    try:
        parquet_path, output_dir, delete_after, show_progress = args
        return process_episode_safe(parquet_path, output_dir, delete_after, show_progress)
    except Exception as e:
        print(f"Critical error in worker process: {str(e)}")
        return False, 0, 0

def main():
    parser = argparse.ArgumentParser(description='Extract images and metadata from ALOHA parquet files (SAFE VERSION)')
    parser.add_argument('--input-dir', default='data/chunk-000', 
                       help='Directory containing episode parquet files')
    parser.add_argument('--output-dir', default='extracted_data',
                       help='Output directory for extracted data')
    parser.add_argument('--keep-parquet', action='store_true',
                       help='Keep original parquet files (do not delete)')
    parser.add_argument('--test-one', action='store_true',
                       help='Process only one episode for testing')
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() // 2),  # Use fewer workers
                       help='Number of parallel workers (default: CPU count / 2)')
    parser.add_argument('--sequential', action='store_true',
                       help='Process episodes sequentially (disable multiprocessing)')
    parser.add_argument('--max-memory-percent', type=float, default=80.0,
                       help='Stop processing if memory usage exceeds this percentage')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all episode parquet files
    parquet_files = sorted(input_dir.glob('episode_*.parquet'))
    
    if not parquet_files:
        print(f"No episode parquet files found in {input_dir}")
        return
    
    if args.test_one:
        parquet_files = parquet_files[:1]
        print(f"Test mode: processing only {parquet_files[0].name}")
    
    print(f"Found {len(parquet_files)} episode files to process")
    print(f"Output directory: {output_dir}")
    print(f"Delete parquet after processing: {not args.keep_parquet}")
    print(f"Max workers: {args.workers}")
    
    # Show system info
    memory = psutil.virtual_memory()
    print(f"Available memory: {memory.available / 1024**3:.1f} GB ({memory.percent:.1f}% used)")
    print()
    
    # Force sequential processing if only 1 worker or few files
    if args.workers == 1 or len(parquet_files) <= 2:
        args.sequential = True
    
    if not args.sequential:
        print(f"Using {args.workers} parallel workers")
    else:
        print("Using sequential processing")
    print()
    
    # Prepare arguments for processing
    process_args = [
        (pf, output_dir, not args.keep_parquet, args.sequential) 
        for pf in parquet_files
    ]
    
    # Process episodes
    successful = 0
    failed = 0
    total_frames = 0
    total_images = 0
    start_time = time.time()
    
    if args.sequential:
        # Sequential processing with detailed progress per episode
        for i, pf_args in enumerate(process_args):
            parquet_path = pf_args[0]
            episode_num = extract_episode_number(parquet_path.name)
            
            print(f"\n[{i+1}/{len(process_args)}] Episode {episode_num}")
            
            # Check system resources before each episode
            if not check_system_resources():
                print(f"Stopping due to resource constraints. Processed {successful} episodes successfully.")
                break
            
            success, frames, images = process_episode_wrapper(pf_args)
            if success:
                successful += 1
                total_frames += frames
                total_images += images
            else:
                failed += 1
                
            # Show progress
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(process_args) - i - 1) / rate if rate > 0 else 0
            
            print(f"Progress: {i+1}/{len(process_args)} | Success: {successful} | Failed: {failed}")
            print(f"Rate: {rate:.2f} episodes/sec | ETA: {eta/60:.1f} minutes")
    else:
        # Parallel processing with resource monitoring
        print("Starting parallel processing...")
        
        # Use fewer workers and disable individual progress bars
        actual_workers = min(args.workers, len(parquet_files))
        process_args_parallel = [
            (pf, output_dir, not args.keep_parquet, False)  # show_progress = False
            for pf in parquet_files
        ]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=actual_workers) as executor:
            # Submit jobs in batches to avoid overwhelming the system
            batch_size = actual_workers * 2
            
            for batch_start in range(0, len(process_args_parallel), batch_size):
                batch_end = min(batch_start + batch_size, len(process_args_parallel))
                batch_args = process_args_parallel[batch_start:batch_end]
                
                # Submit batch
                future_to_episode = {
                    executor.submit(process_episode_wrapper, pf_args): extract_episode_number(pf_args[0].name)
                    for pf_args in batch_args
                }
                
                # Wait for batch to complete
                for future in concurrent.futures.as_completed(future_to_episode):
                    episode_num = future_to_episode[future]
                    
                    try:
                        success, frames, images = future.result(timeout=300)  # 5 minute timeout per episode
                        if success:
                            successful += 1
                            total_frames += frames
                            total_images += images
                        else:
                            failed += 1
                            
                    except concurrent.futures.TimeoutError:
                        print(f"Episode {episode_num} timed out")
                        failed += 1
                    except Exception as exc:
                        print(f"Episode {episode_num} failed: {str(exc)}")
                        failed += 1
                
                # Check resources between batches
                if not check_system_resources():
                    print(f"Stopping due to resource constraints after {successful + failed} episodes")
                    break
                
                print(f"Batch complete. Success: {successful}, Failed: {failed}")
    
    total_time = time.time() - start_time
    
    print(f"\n" + "="*60)
    print(f"Processing complete!")
    print(f"Successfully processed: {successful} episodes")
    print(f"Failed: {failed} episodes")
    print(f"Total frames processed: {total_frames:,}")
    print(f"Total images saved: {total_images:,}")
    print(f"Total time: {total_time/60:.1f} minutes")
    
    if successful > 0:
        print(f"\nExtracted data saved to: {output_dir}")
        print("\nEach episode directory contains:")
        print("  - PNG images: episode_episodenum_framenum_cam_name.png")
        print("  - JSON metadata: episode_episodenum_framenum.json")
        
        # Show example of what was created
        first_episode_dir = next(output_dir.glob('episode_*'), None)
        if first_episode_dir:
            png_files = list(first_episode_dir.glob('*.png'))
            meta_files = list(first_episode_dir.glob('*.json'))
            
            print(f"\nExample from {first_episode_dir.name}:")
            print(f"  {len(png_files)} PNG files")
            print(f"  {len(meta_files)} metadata files")

if __name__ == "__main__":
    main()