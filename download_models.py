#!/usr/bin/env python3
"""
Kanibus Model Downloader - WAN Compatible (2025)
================================================

Automatically downloads modern WAN-compatible models for Kanibus system.
Supports both T2I-Adapters (recommended) and legacy ControlNet models.

Usage:
    python download_models.py                    # Download recommended T2I-Adapters
    python download_models.py --legacy           # Download legacy ControlNet models
    python download_models.py --all              # Download all models (T2I + video + legacy)
    python download_models.py --comfyui-path /path/to/ComfyUI
    python download_models.py --check-only       # Just check if models exist
    python download_models.py --video-only       # Download only video-specific models
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

# Modern T2I-Adapters (RECOMMENDED - 94% more efficient)
T2I_ADAPTER_MODELS = {
    "t2iadapter_sketch_sd14v1.pth": {
        "url": "https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_sketch_sd14v1.pth",
        "size_mb": 158,
        "description": "T2I-Adapter Sketch - For eye mask control (94% smaller than ControlNet)",
        "directory": "t2i_adapter",
        "priority": "high"
    },
    "t2iadapter_depth_sd14v1.pth": {
        "url": "https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_depth_sd14v1.pth",
        "size_mb": 158,
        "description": "T2I-Adapter Depth - For depth map control (94% smaller than ControlNet)",
        "directory": "t2i_adapter",
        "priority": "high"
    },
    "t2iadapter_canny_sd14v1.pth": {
        "url": "https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_canny_sd14v1.pth",
        "size_mb": 158,
        "description": "T2I-Adapter Canny - For edge detection control",
        "directory": "t2i_adapter",
        "priority": "high"
    },
    "t2iadapter_openpose_sd14v1.pth": {
        "url": "https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_openpose_sd14v1.pth",
        "size_mb": 158,
        "description": "T2I-Adapter OpenPose - For pose control",
        "directory": "t2i_adapter",
        "priority": "high"
    }
}

# Video-specific models for WAN compatibility
VIDEO_MODELS = {
    "svd_controlnet.safetensors": {
        "url": "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_controlnet.safetensors",
        "size_mb": 2100,
        "description": "SVD ControlNet - For video temporal control with WAN",
        "directory": "controlnet",
        "priority": "medium"
    },
    "i2v_adapter.safetensors": {
        "url": "https://huggingface.co/TencentARC/I2V-Adapter/resolve/main/i2v_adapter.safetensors",
        "size_mb": 850,
        "description": "I2V-Adapter - Image-to-Video conversion with control",
        "directory": "controlnet",
        "priority": "medium"
    }
}

# Legacy ControlNet models (for backward compatibility)
LEGACY_CONTROLNET_MODELS = {
    "control_v11p_sd15_scribble.pth": {
        "url": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble.pth",
        "size_mb": 1400,
        "description": "ControlNet Scribble - LEGACY (Use T2I-Adapter instead)",
        "directory": "controlnet",
        "priority": "low"
    },
    "control_v11f1p_sd15_depth.pth": {
        "url": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth",
        "size_mb": 1400,
        "description": "ControlNet Depth - LEGACY (Use T2I-Adapter instead)",
        "directory": "controlnet",
        "priority": "low"
    },
    "control_v11p_sd15_normalbae.pth": {
        "url": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_normalbae.pth",
        "size_mb": 1400,
        "description": "ControlNet Normal - LEGACY (Use T2I-Adapter Canny instead)",
        "directory": "controlnet",
        "priority": "low"
    },
    "control_v11p_sd15_openpose.pth": {
        "url": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth",
        "size_mb": 1400,
        "description": "ControlNet OpenPose - LEGACY (Use T2I-Adapter instead)",
        "directory": "controlnet",
        "priority": "low"
    }
}

# Combined model sets for different download modes
REQUIRED_MODELS = T2I_ADAPTER_MODELS  # Default to modern T2I-Adapters

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

def get_model_path(comfyui_path: Path, model_type: str = "controlnet") -> Path:
    """Get models directory for specified type"""
    return comfyui_path / "models" / model_type

def get_controlnet_path(comfyui_path: Path) -> Path:
    """Get ControlNet models directory (legacy compatibility)"""
    return get_model_path(comfyui_path, "controlnet")

def get_t2i_adapter_path(comfyui_path: Path) -> Path:
    """Get T2I-Adapter models directory"""
    return get_model_path(comfyui_path, "t2i_adapter")

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

def check_installation(comfyui_path: Path, model_dict: Dict) -> Dict[str, bool]:
    """Check which models are already installed"""
    results = {}
    
    print_header("CHECKING EXISTING MODELS")
    
    for model_name, model_info in model_dict.items():
        model_dir = model_info.get("directory", "controlnet")
        model_path = get_model_path(comfyui_path, model_dir) / model_name
        exists = check_model_exists(model_path, model_info["size_mb"])
        results[model_name] = exists
        
        status_icon = "✅" if exists else "⚠️"
        priority_icon = "🔥" if model_info.get("priority") == "high" else "📦"
        print(f"{status_icon} {priority_icon} {model_name} - {'Already installed' if exists else 'Not found or invalid'}")
        if not exists:
            print(f"    📁 Expected location: {model_path}")
    
    return results

def download_missing_models(comfyui_path: Path, model_dict: Dict, force_download: bool = False) -> bool:
    """Download all missing models from specified dictionary"""
    
    # Check existing models
    existing_models = check_installation(comfyui_path, model_dict)
    
    # Determine what to download
    to_download = []
    for model_name, model_info in model_dict.items():
        if force_download or not existing_models.get(model_name, False):
            to_download.append((model_name, model_info))
    
    if not to_download:
        print_success("All models are already installed!")
        return True
    
    print_header("DOWNLOADING MISSING MODELS")
    
    total_size_mb = sum(info["size_mb"] for _, info in to_download)
    total_size_gb = total_size_mb / 1024
    print_info(f"Will download {len(to_download)} models (~{total_size_gb:.1f}GB)")
    
    # Show what will be downloaded
    for model_name, model_info in to_download:
        priority_icon = "🔥" if model_info.get("priority") == "high" else "📦"
        print(f"  {priority_icon} {model_name} ({model_info['size_mb']}MB) - {model_info['description']}")
    
    # Confirm download
    if not force_download:
        response = input(f"\nProceed with download? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print_info("Download cancelled by user")
            return False
    
    # Download each model
    success_count = 0
    for model_name, model_info in to_download:
        model_dir = model_info.get("directory", "controlnet")
        model_path = get_model_path(comfyui_path, model_dir) / model_name
        
        # Create directory if it doesn't exist
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'-'*60}")
        priority_text = f"[{model_info.get('priority', 'normal').upper()}]" 
        print_info(f"Downloading {model_name} {priority_text} ({model_info['size_mb']}MB)")
        print_info(f"Description: {model_info['description']}")
        print_info(f"Directory: {model_dir}/")
        
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

def get_model_set(args) -> Dict:
    """Determine which model set to use based on arguments"""
    if args.all:
        # Combine all model types
        combined = {}
        combined.update(T2I_ADAPTER_MODELS)
        combined.update(VIDEO_MODELS)
        combined.update(LEGACY_CONTROLNET_MODELS)
        return combined
    elif args.legacy:
        return LEGACY_CONTROLNET_MODELS
    elif args.video_only:
        return VIDEO_MODELS
    else:
        # Default: T2I-Adapters (recommended)
        return T2I_ADAPTER_MODELS

def print_model_info():
    """Print information about available model types"""
    print_header("AVAILABLE MODEL TYPES")
    
    print_info("🔥 T2I-ADAPTERS (Recommended)")
    print("   • 94% smaller than ControlNet (158MB vs 1.4GB each)")
    print("   • 93.69% fewer parameters")
    print("   • Near-zero impact on generation speed")
    print("   • Optimized for WAN 2.1/2.2 compatibility")
    print("   • Total size: ~632MB\n")
    
    print_info("🎬 VIDEO MODELS")
    print("   • SVD ControlNet for temporal video control")
    print("   • I2V-Adapter for image-to-video conversion")
    print("   • Optimized for video workflows")
    print("   • Total size: ~2.95GB\n")
    
    print_info("📦 LEGACY CONTROLNET (Backward Compatibility)")
    print("   • Original SD1.5-based ControlNet models")
    print("   • Larger file sizes (1.4GB each)")
    print("   • Slower processing")
    print("   • Total size: ~5.6GB")
    print("   • Use only if T2I-Adapters don't work\n")

def main():
    parser = argparse.ArgumentParser(
        description="Download WAN-compatible models for Kanibus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python download_models.py                 # Download T2I-Adapters (recommended)
  python download_models.py --legacy        # Download legacy ControlNet models
  python download_models.py --all           # Download all model types
  python download_models.py --video-only    # Download only video models
  python download_models.py --check-only    # Just check existing models"""
    )
    
    parser.add_argument("--comfyui-path", type=str, help="Path to ComfyUI installation")
    parser.add_argument("--check-only", action="store_true", help="Only check existing models")
    parser.add_argument("--force", action="store_true", help="Force re-download all models")
    parser.add_argument("--legacy", action="store_true", help="Download legacy ControlNet models")
    parser.add_argument("--all", action="store_true", help="Download all models (T2I + video + legacy)")
    parser.add_argument("--video-only", action="store_true", help="Download only video-specific models")
    parser.add_argument("--info", action="store_true", help="Show information about model types")
    
    args = parser.parse_args()
    
    if args.info:
        print_model_info()
        return 0
    
    print_header("KANIBUS MODEL DOWNLOADER - WAN COMPATIBLE (2025)")
    
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
    
    # Determine which model set to use
    model_dict = get_model_set(args)
    
    # Show selected model type
    if args.all:
        print_info("📦 Selected: ALL MODELS (T2I-Adapters + Video + Legacy)")
    elif args.legacy:
        print_warning("📦 Selected: LEGACY CONTROLNET MODELS (Consider using T2I-Adapters instead)")
    elif args.video_only:
        print_info("🎬 Selected: VIDEO-SPECIFIC MODELS")
    else:
        print_success("🔥 Selected: T2I-ADAPTERS (Recommended for WAN compatibility)")
    
    # Create required directories
    directories_to_create = set()
    for model_info in model_dict.values():
        directories_to_create.add(model_info.get("directory", "controlnet"))
    
    for directory in directories_to_create:
        dir_path = get_model_path(comfyui_path, directory)
        if not dir_path.exists():
            print_info(f"Creating directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # Check existing models
    existing_models = check_installation(comfyui_path, model_dict)
    
    if args.check_only:
        missing = [name for name, exists in existing_models.items() if not exists]
        if missing:
            print_warning(f"Missing models: {len(missing)}/{len(model_dict)}")
            for name in missing:
                print(f"  ❌ {name}")
            return 1
        else:
            print_success("All required models are installed!")
            return 0
    
    # Download missing models
    success = download_missing_models(comfyui_path, model_dict, args.force)
    
    if success:
        print_header("INSTALLATION COMPLETE")
        print_success("Kanibus is ready to use with modern WAN-compatible models!")
        if not args.legacy:
            print_info("🔥 You're using the efficient T2I-Adapter models")
            print_info("   • 94% smaller than legacy ControlNet")
            print_info("   • Optimized for WAN 2.1/2.2 compatibility")
        print_info("Restart ComfyUI to load the new models")
        return 0
    else:
        print_header("INSTALLATION FAILED")
        print_error("Some models failed to download")
        print_info("Please check your internet connection and try again")
        return 1

if __name__ == "__main__":
    sys.exit(main())