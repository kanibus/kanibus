#!/usr/bin/env python3
"""
Kanibus Model Downloader
========================

Automatically downloads required ControlNet models for Kanibus system.
Run this script after installing Kanibus to download all necessary models.

Usage:
    python download_models.py
    python download_models.py --comfyui-path /path/to/ComfyUI
    python download_models.py --check-only  # Just check if models exist
"""

import os
import sys
import requests
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from tqdm import tqdm

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_success(msg: str):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")

def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠️ {msg}{Colors.END}")

def print_error(msg: str):
    print(f"{Colors.RED}❌ {msg}{Colors.END}")

def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ️ {msg}{Colors.END}")

def print_header(msg: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{msg.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

# Required ControlNet models
REQUIRED_MODELS = {
    "control_v11p_sd15_scribble.pth": {
        "url": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble.pth",
        "size_mb": 1400,
        "description": "ControlNet Scribble - For eye mask control",
        "sha256": None  # Add hash if available
    },
    "control_v11f1p_sd15_depth.pth": {
        "url": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth",
        "size_mb": 1400,
        "description": "ControlNet Depth - For depth map control",
        "sha256": None
    },
    "control_v11p_sd15_normalbae.pth": {
        "url": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_normalbae.pth",
        "size_mb": 1400,
        "description": "ControlNet Normal - For normal map control",
        "sha256": None
    },
    "control_v11p_sd15_openpose.pth": {
        "url": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth",
        "size_mb": 1400,
        "description": "ControlNet OpenPose - For pose control",
        "sha256": None
    }
}

def find_comfyui_path() -> Optional[Path]:
    """Find ComfyUI installation path"""
    possible_paths = [
        # Current directory structure
        Path(__file__).parent.parent.parent.parent,
        # Common installation paths
        Path.home() / "ComfyUI",
        Path("ComfyUI"),
        Path("../../../.."),  # Relative from custom_nodes/Kanibus
        # Windows common paths
        Path("C:/ComfyUI"),
        Path("D:/ComfyUI"),
        # Linux/Mac common paths
        Path("/opt/ComfyUI"),
        Path("/usr/local/ComfyUI"),
    ]
    
    for path in possible_paths:
        if path.exists() and (path / "main.py").exists():
            return path.resolve()
    
    return None

def get_controlnet_path(comfyui_path: Path) -> Path:
    """Get ControlNet models directory"""
    return comfyui_path / "models" / "controlnet"

def check_model_exists(model_path: Path, expected_size_mb: int) -> bool:
    """Check if model exists and has reasonable size"""
    if not model_path.exists():
        return False
    
    # Check file size (allow 10% variance)
    actual_size_mb = model_path.stat().st_size / (1024 * 1024)
    min_size = expected_size_mb * 0.9
    max_size = expected_size_mb * 1.1
    
    return min_size <= actual_size_mb <= max_size

def download_file(url: str, destination: Path, description: str) -> bool:
    """Download file with progress bar"""
    try:
        print_info(f"Downloading {description}...")
        print_info(f"Source: {url}")
        print_info(f"Destination: {destination}")
        
        # Create directory if it doesn't exist
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f:
            with tqdm(
                desc=description,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print_success(f"Downloaded {description}")
        return True
        
    except requests.RequestException as e:
        print_error(f"Failed to download {description}: {e}")
        return False
    except Exception as e:
        print_error(f"Unexpected error downloading {description}: {e}")
        return False

def verify_download(file_path: Path, expected_hash: Optional[str] = None) -> bool:
    """Verify downloaded file integrity"""
    if not file_path.exists():
        return False
    
    # Basic size check
    if file_path.stat().st_size < 1024 * 1024:  # Less than 1MB is suspicious
        print_warning(f"File {file_path.name} seems too small")
        return False
    
    # Hash verification if provided
    if expected_hash:
        print_info(f"Verifying hash for {file_path.name}...")
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        if sha256_hash.hexdigest() != expected_hash:
            print_error(f"Hash mismatch for {file_path.name}")
            return False
        
        print_success(f"Hash verified for {file_path.name}")
    
    return True

def check_installation(comfyui_path: Path) -> Dict[str, bool]:
    """Check which models are already installed"""
    controlnet_path = get_controlnet_path(comfyui_path)
    results = {}
    
    print_header("CHECKING EXISTING MODELS")
    
    for model_name, model_info in REQUIRED_MODELS.items():
        model_path = controlnet_path / model_name
        exists = check_model_exists(model_path, model_info["size_mb"])
        results[model_name] = exists
        
        if exists:
            print_success(f"{model_name} - Already installed")
        else:
            print_warning(f"{model_name} - Not found or invalid")
    
    return results

def download_missing_models(comfyui_path: Path, force_download: bool = False) -> bool:
    """Download all missing models"""
    controlnet_path = get_controlnet_path(comfyui_path)
    
    # Check existing models
    existing_models = check_installation(comfyui_path)
    
    # Determine what to download
    to_download = []
    for model_name, model_info in REQUIRED_MODELS.items():
        if force_download or not existing_models.get(model_name, False):
            to_download.append((model_name, model_info))
    
    if not to_download:
        print_success("All models are already installed!")
        return True
    
    print_header("DOWNLOADING MISSING MODELS")
    
    total_size_mb = sum(info["size_mb"] for _, info in to_download)
    print_info(f"Will download {len(to_download)} models (~{total_size_mb/1024:.1f}GB)")
    
    # Confirm download
    if not force_download:
        response = input(f"\nProceed with download? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print_info("Download cancelled by user")
            return False
    
    # Download each model
    success_count = 0
    for model_name, model_info in to_download:
        model_path = controlnet_path / model_name
        
        print(f"\n{'-'*60}")
        print_info(f"Downloading {model_name} ({model_info['size_mb']}MB)")
        print_info(f"Description: {model_info['description']}")
        
        if download_file(model_info["url"], model_path, model_name):
            if verify_download(model_path, model_info.get("sha256")):
                success_count += 1
            else:
                print_error(f"Verification failed for {model_name}")
                # Remove corrupted file
                if model_path.exists():
                    model_path.unlink()
        
        print(f"{'-'*60}")
    
    print_header("DOWNLOAD SUMMARY")
    print_info(f"Successfully downloaded: {success_count}/{len(to_download)} models")
    
    if success_count == len(to_download):
        print_success("All models downloaded successfully!")
        return True
    else:
        print_error(f"Failed to download {len(to_download) - success_count} models")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download required Kanibus models")
    parser.add_argument("--comfyui-path", type=str, help="Path to ComfyUI installation")
    parser.add_argument("--check-only", action="store_true", help="Only check existing models")
    parser.add_argument("--force", action="store_true", help="Force re-download all models")
    
    args = parser.parse_args()
    
    print_header("KANIBUS MODEL DOWNLOADER")
    
    # Find ComfyUI path
    if args.comfyui_path:
        comfyui_path = Path(args.comfyui_path)
    else:
        comfyui_path = find_comfyui_path()
    
    if not comfyui_path or not comfyui_path.exists():
        print_error("Could not find ComfyUI installation!")
        print_info("Please specify path with --comfyui-path /path/to/ComfyUI")
        return 1
    
    print_success(f"Found ComfyUI at: {comfyui_path}")
    
    # Check if controlnet directory exists
    controlnet_path = get_controlnet_path(comfyui_path)
    if not controlnet_path.exists():
        print_info(f"Creating ControlNet directory: {controlnet_path}")
        controlnet_path.mkdir(parents=True, exist_ok=True)
    
    # Check existing models
    existing_models = check_installation(comfyui_path)
    
    if args.check_only:
        missing = [name for name, exists in existing_models.items() if not exists]
        if missing:
            print_warning(f"Missing models: {', '.join(missing)}")
            return 1
        else:
            print_success("All required models are installed!")
            return 0
    
    # Download missing models
    success = download_missing_models(comfyui_path, args.force)
    
    if success:
        print_header("INSTALLATION COMPLETE")
        print_success("Kanibus is ready to use!")
        print_info("Restart ComfyUI to load the new models")
        return 0
    else:
        print_header("INSTALLATION FAILED")
        print_error("Some models failed to download")
        print_info("Please check your internet connection and try again")
        return 1

if __name__ == "__main__":
    sys.exit(main())