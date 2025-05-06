#!/usr/bin/env python3
"""
Directory Scanner Script

This script scans a target directory and generates a YAML file containing information
about all folders and files in the directory structure. The YAML file will be used
for further security analysis.
"""

import os
import yaml
import argparse
import uuid
import hashlib
import datetime
from pathlib import Path
import mimetypes
import re
import time

def calculate_file_hash(file_path, skip_hash=False):
    """
    Calculate the SHA-256 hash of a file.
    
    Args:
        file_path (str): Path to the file
        skip_hash (bool): If True, skip hash calculation
        
    Returns:
        str: Hexadecimal digest of the hash, or "skipped" if skip_hash is True
    """
    if skip_hash:
        return "skipped"
        
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read the file in chunks to handle large files efficiently
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        print(f"Warning: Could not calculate hash for {file_path}: {e}")
        return "hash_error"

def parse_gitignore(gitignore_path):
    """
    Parse a .gitignore file and return a list of patterns.
    
    Args:
        gitignore_path (str): Path to the .gitignore file
        
    Returns:
        list: List of patterns from the .gitignore file
    """
    patterns = []
    try:
        with open(gitignore_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    patterns.append(line)
    except Exception as e:
        print(f"Warning: Could not parse .gitignore file {gitignore_path}: {e}")
    
    return patterns

def is_ignored(file_path, rel_path, gitignore_patterns):
    """
    Check if a file or directory should be ignored based on gitignore patterns.
    
    Args:
        file_path (str): Absolute path to the file or directory
        rel_path (str): Relative path from the target directory
        gitignore_patterns (list): List of patterns from .gitignore
        
    Returns:
        bool: True if the file should be ignored, False otherwise
    """
    if not gitignore_patterns:
        return False
    
    # Prepare the path for matching (ensure it starts with ./ for some patterns)
    check_path = os.path.join('.', rel_path) if rel_path else '.'
    
    for pattern in gitignore_patterns:
        # Handle negation patterns (those starting with !)
        is_negation = pattern.startswith('!')
        if is_negation:
            pattern = pattern[1:]
        
        # Convert gitignore pattern to regex pattern
        regex_pattern = pattern
        regex_pattern = regex_pattern.replace('.', '\\.')
        regex_pattern = regex_pattern.replace('*', '.*')
        regex_pattern = regex_pattern.replace('?', '.')
        
        # Handle directory indicators (trailing slash)
        is_dir_only = regex_pattern.endswith('/')
        if is_dir_only:
            regex_pattern = regex_pattern[:-1]
        
        # Handle patterns with no slashes (match anywhere in path)
        if '/' not in regex_pattern:
            regex_pattern = f".*{regex_pattern}"
        
        # Check if pattern matches
        if re.match(f"^{regex_pattern}$", check_path):
            return not is_negation
    
    return False

def scan_directory(target_dir, output_file, ignore_patterns=None, use_gitignore=True, skip_hash=False):
    """
    Scans a directory recursively and generates a YAML file with file information.
    
    Args:
        target_dir (str): Path to the directory to scan
        output_file (str): Path to the output YAML file
        ignore_patterns (list): List of patterns to ignore (e.g., ['.git', '__pycache__'])
        use_gitignore (bool): Whether to respect .gitignore rules
        skip_hash (bool): Whether to skip hash calculation for performance
    """
    if ignore_patterns is None:
        ignore_patterns = ['.git', '__pycache__', 'venv', 'node_modules', '.DS_Store']
    
    # Check for .gitignore file
    gitignore_path = os.path.join(target_dir, '.gitignore')
    gitignore_patterns = []
    if use_gitignore and os.path.isfile(gitignore_path):
        print(f"Found .gitignore file at {gitignore_path}")
        gitignore_patterns = parse_gitignore(gitignore_path)
        print(f"Parsed {len(gitignore_patterns)} patterns from .gitignore")
        
    # Start timestamp for scan
    start_time = time.time()

    # Initialize the data structure
    result = {
        'scan_info': {
            'target_directory': os.path.abspath(target_dir),
            'total_folders': 0,
            'total_files': 0
        },
        'folders': []
    }
    
    # Track unique IDs to avoid collisions
    used_ids = set()
    
    # Walk through the directory
    for root, dirs, files in os.walk(target_dir):
        # Filter out directories to ignore
        dirs[:] = [d for d in dirs if not any(pattern in d for pattern in ignore_patterns)]
        
        # Skip ignored directories
        if any(pattern in root for pattern in ignore_patterns):
            continue
        
        # Create relative path from target directory
        rel_path = os.path.relpath(root, target_dir)
        if rel_path == '.':
            rel_path = ''
        
        # Create folder entry
        folder_entry = {
            'path': rel_path,
            'files': []
        }
        
        # Process files in this folder
        for filename in files:
            # Skip files matching ignore patterns
            if any(pattern in filename for pattern in ignore_patterns):
                continue
                
            file_path = os.path.join(root, filename)
            file_rel_path = os.path.join(rel_path, filename) if rel_path else filename
            
            # Check if file is ignored by .gitignore
            if is_ignored(file_path, file_rel_path, gitignore_patterns):
                continue
            
            # Generate a unique ID for the file
            file_id = str(uuid.uuid4())
            while file_id in used_ids:
                file_id = str(uuid.uuid4())
            used_ids.add(file_id)
            
            # Get file type using mimetypes
            file_extension = os.path.splitext(filename)[1]
            file_type = mimetypes.guess_type(filename)[0] or f"unknown{file_extension}"
            
            # Get file metadata
            try:
                file_stats = os.stat(file_path)
                created_time = datetime.datetime.fromtimestamp(file_stats.st_ctime).isoformat()
                modified_time = datetime.datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                file_size = file_stats.st_size
                
                # Calculate file hash
                file_hash = calculate_file_hash(file_path, skip_hash)
                
                # Create file entry
                file_entry = {
                    'id': file_id,
                    'folder': rel_path,
                    'file_name': filename,
                    'file_type': file_type,
                    'metadata': {
                        'created_date': created_time,
                        'last_modified': modified_time,
                        'last_reviewed': '',  # To be filled during LLM processing
                        'size_bytes': file_size,
                        'hash': file_hash
                    },
                    'description': '',  # To be filled by LLM
                    'security_observations': ''  # To be filled by LLM
                }
                
                # Progress update for large repos
                files_processed = sum(len(folder['files']) for folder in result['folders']) + len(folder_entry['files'])
                if files_processed % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"Processed {files_processed} files so far (elapsed: {elapsed:.2f}s)")
                
            except Exception as e:
                print(f"Warning: Error processing file {file_path}: {e}")
                # Create minimal file entry for files with errors
                file_entry = {
                    'id': file_id,
                    'folder': rel_path,
                    'file_name': filename,
                    'file_type': file_type,
                    'metadata': {
                        'created_date': '',
                        'last_modified': '',
                        'last_reviewed': '',
                        'size_bytes': 0,
                        'hash': 'error'
                    },
                    'description': 'Error processing file',
                    'security_observations': 'File could not be processed, may require manual review'
                }
            
            folder_entry['files'].append(file_entry)
        
        # Only add folders that have files
        if folder_entry['files']:
            result['folders'].append(folder_entry)
            result['scan_info']['total_folders'] += 1
            result['scan_info']['total_files'] += len(folder_entry['files'])
    
    # Add scan completion time to scan_info
    scan_duration = time.time() - start_time
    result['scan_info']['scan_duration_seconds'] = round(scan_duration, 2)
    result['scan_info']['scan_timestamp'] = datetime.datetime.now().isoformat()
    
    # Write to YAML file
    with open(output_file, 'w') as f:
        yaml.dump(result, f, default_flow_style=False, sort_keys=False)
    
    print(f"Scan completed in {scan_duration:.2f} seconds.")
    print(f"Found {result['scan_info']['total_folders']} folders and {result['scan_info']['total_files']} files.")
    print(f"Results written to {output_file}")
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Scan directory structure and generate YAML file for security analysis')
    parser.add_argument('target_dir', help='Target directory to scan')
    parser.add_argument('--output', '-o', default='directory_scan.yaml', help='Output YAML file (default: directory_scan.yaml)')
    parser.add_argument('--ignore', '-i', nargs='+', help='Additional patterns to ignore (e.g., .git __pycache__)')
    parser.add_argument('--no-gitignore', action='store_true', help='Ignore .gitignore file even if present')
    parser.add_argument('--no-hash', action='store_true', help='Skip file hash calculation (faster but less comprehensive)')
    
    args = parser.parse_args()
    
    # Ensure target directory exists
    if not os.path.isdir(args.target_dir):
        print(f"Error: Target directory '{args.target_dir}' does not exist.")
        return 1
    
    # Display scan start message
    print(f"Starting scan of '{os.path.abspath(args.target_dir)}'")
    print(f"Output will be written to '{args.output}'")
    
    try:
        # Perform the scan
        scan_directory(
            args.target_dir, 
            args.output, 
            args.ignore,
            use_gitignore=not args.no_gitignore,
            skip_hash=args.no_hash
        )
    except KeyboardInterrupt:
        print("\nScan interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nError during scan: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    main()