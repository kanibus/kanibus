#!/usr/bin/env python3
"""
Kanibus Installation Test
========================

Tests if Kanibus is properly installed with all required models and dependencies.

Usage:
    python test_installation.py
    python test_installation.py --verbose
    python test_installation.py --fix-issues  # Attempt to fix common problems
"""

import os
import sys
import importlib
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_test_result(test_name: str, passed: bool, details: str = ""):
    status = f"{Colors.GREEN}✅ PASS{Colors.END}" if passed else f"{Colors.RED}❌ FAIL{Colors.END}"
    print(f"{status} {test_name}")
    if details and not passed:
        print(f"    {Colors.YELLOW}Details: {details}{Colors.END}")

def print_header(title: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def test_python_version() -> Tuple[bool, str]:
    """Test Python version compatibility"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)"

def test_torch_installation() -> Tuple[bool, str]:
    """Test PyTorch installation and GPU support"""
    try:
        import torch
        version = torch.__version__
        
        # Test CUDA availability
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            return True, f"PyTorch {version}, CUDA available, {gpu_count} GPU(s), {gpu_name}"
        else:
            # Test MPS (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return True, f"PyTorch {version}, MPS (Apple Silicon) available"
            else:
                return True, f"PyTorch {version}, CPU only"
    
    except ImportError:
        return False, "PyTorch not installed"
    except Exception as e:
        return False, f"PyTorch error: {str(e)}"

def test_core_dependencies() -> Dict[str, Tuple[bool, str]]:
    """Test core dependencies"""
    dependencies = {
        "mediapipe": "MediaPipe for face/pose detection",
        "cv2": "OpenCV for computer vision",
        "numpy": "NumPy for numerical computing", 
        "PIL": "Pillow for image processing",
        "transformers": "Hugging Face Transformers",
        "ultralytics": "YOLO models",
        "scipy": "Scientific computing",
        "sklearn": "Machine learning utilities",
    }
    
    results = {}
    
    for package, description in dependencies.items():
        try:
            if package == "cv2":
                import cv2
                version = cv2.__version__
            elif package == "PIL":
                from PIL import Image
                version = Image.__version__ if hasattr(Image, '__version__') else "Unknown"
            else:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'Unknown')
            
            results[package] = (True, f"{description} v{version}")
        
        except ImportError:
            results[package] = (False, f"{description} - Not installed")
        except Exception as e:
            results[package] = (False, f"{description} - Error: {str(e)}")
    
    return results

def test_mediapipe_functionality() -> Tuple[bool, str]:
    """Test MediaPipe face detection functionality"""
    try:
        import mediapipe as mp
        import numpy as np
        
        # Initialize MediaPipe Face Mesh
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Create dummy image
        dummy_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Process dummy image (should not crash)
        results = face_mesh.process(dummy_image)
        
        return True, "MediaPipe Face Mesh initialized successfully"
    
    except Exception as e:
        return False, f"MediaPipe test failed: {str(e)}"

def find_comfyui_path() -> Optional[Path]:
    """Find ComfyUI installation path"""
    possible_paths = [
        Path(__file__).parent.parent.parent.parent,
        Path.home() / "ComfyUI",
        Path("ComfyUI"),
        Path("../../../.."),
        Path("C:/ComfyUI"),
        Path("D:/ComfyUI"),
        Path("/opt/ComfyUI"),
        Path("/usr/local/ComfyUI"),
    ]
    
    for path in possible_paths:
        if path.exists() and (path / "main.py").exists():
            return path.resolve()
    
    return None

def test_controlnet_models() -> Dict[str, Tuple[bool, str]]:
    """Test if required ControlNet models are installed"""
    required_models = {
        "control_v11p_sd15_scribble.pth": "Scribble control for eye masks",
        "control_v11f1p_sd15_depth.pth": "Depth control for depth maps",
        "control_v11p_sd15_normalbae.pth": "Normal control for surface normals",
        "control_v11p_sd15_openpose.pth": "Pose control for body pose",
    }
    
    results = {}
    
    # Find ComfyUI path
    comfyui_path = find_comfyui_path()
    if not comfyui_path:
        for model_name, description in required_models.items():
            results[model_name] = (False, f"{description} - ComfyUI path not found")
        return results
    
    controlnet_path = comfyui_path / "models" / "controlnet"
    
    for model_name, description in required_models.items():
        model_path = controlnet_path / model_name
        
        if model_path.exists():
            # Check file size (should be around 1.4GB)
            size_mb = model_path.stat().st_size / (1024 * 1024)
            if size_mb > 100:  # At least 100MB
                results[model_name] = (True, f"{description} ({size_mb:.0f}MB)")
            else:
                results[model_name] = (False, f"{description} - File too small ({size_mb:.0f}MB)")
        else:
            results[model_name] = (False, f"{description} - File not found")
    
    return results

def test_kanibus_nodes() -> Dict[str, Tuple[bool, str]]:
    """Test if Kanibus nodes can be imported"""
    expected_nodes = [
        "KanibusMaster",
        "VideoFrameLoader", 
        "NeuralPupilTracker",
        "AdvancedTrackingPro",
        "SmartFacialMasking",
        "AIDepthControl",
        "NormalMapGenerator",
        "LandmarkPro468",
        "EmotionAnalyzer",
        "HandTracking",
        "BodyPoseEstimator",
        "ObjectSegmentation",
        "TemporalSmoother",
        "MultiControlNetApply",
    ]
    
    results = {}
    
    try:
        # Add current directory to path to import nodes
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        
        from nodes import NODE_CLASS_MAPPINGS
        
        for node_name in expected_nodes:
            if node_name in NODE_CLASS_MAPPINGS:
                results[node_name] = (True, f"Node class found and importable")
            else:
                results[node_name] = (False, f"Node class not found in mappings")
    
    except ImportError as e:
        for node_name in expected_nodes:
            results[node_name] = (False, f"Import error: {str(e)}")
    except Exception as e:
        for node_name in expected_nodes:
            results[node_name] = (False, f"Error: {str(e)}")
    
    return results

def test_gpu_memory() -> Tuple[bool, str]:
    """Test available GPU memory"""
    try:
        import torch
        
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            free_memory = total_memory - allocated_memory
            
            total_gb = total_memory / (1024**3)
            free_gb = free_memory / (1024**3)
            
            if free_gb >= 4.0:
                return True, f"GPU memory: {free_gb:.1f}GB free / {total_gb:.1f}GB total"
            else:
                return False, f"Low GPU memory: {free_gb:.1f}GB free / {total_gb:.1f}GB total (need 4GB+)"
        
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return True, "Apple Silicon MPS available (unified memory)"
        
        else:
            return True, "CPU only (no GPU memory check needed)"
    
    except Exception as e:
        return False, f"GPU memory check failed: {str(e)}"

def fix_common_issues(verbose: bool = False) -> Dict[str, bool]:
    """Attempt to fix common issues"""
    fixes_applied = {}
    
    print_header("ATTEMPTING FIXES")
    
    # Fix 1: Install missing dependencies
    try:
        print("Checking for missing Python packages...")
        missing_packages = []
        
        dependencies = ["mediapipe", "opencv-python", "transformers", "ultralytics"]
        for package in dependencies:
            try:
                importlib.import_module(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"Installing missing packages: {', '.join(missing_packages)}")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install"
            ] + missing_packages, capture_output=True, text=True)
            
            fixes_applied["install_packages"] = result.returncode == 0
            if verbose and result.returncode != 0:
                print(f"Package installation failed: {result.stderr}")
        else:
            fixes_applied["install_packages"] = True
    
    except Exception as e:
        fixes_applied["install_packages"] = False
        if verbose:
            print(f"Package installation error: {e}")
    
    # Fix 2: Download missing ControlNet models
    try:
        print("Checking for missing ControlNet models...")
        model_results = test_controlnet_models()
        missing_models = [name for name, (exists, _) in model_results.items() if not exists]
        
        if missing_models:
            print(f"Found {len(missing_models)} missing models")
            print("Run: python download_models.py")
            fixes_applied["download_models"] = False  # Can't auto-fix this
        else:
            fixes_applied["download_models"] = True
            
    except Exception as e:
        fixes_applied["download_models"] = False
        if verbose:
            print(f"Model check error: {e}")
    
    return fixes_applied

def run_all_tests(verbose: bool = False) -> Dict[str, Dict[str, Tuple[bool, str]]]:
    """Run all installation tests"""
    all_results = {}
    
    print_header("KANIBUS INSTALLATION TEST")
    
    # Test 1: Python version
    print_header("PYTHON ENVIRONMENT")
    python_result = test_python_version()
    print_test_result("Python Version", python_result[0], python_result[1])
    all_results["python"] = {"version": python_result}
    
    # Test 2: PyTorch
    torch_result = test_torch_installation()
    print_test_result("PyTorch Installation", torch_result[0], torch_result[1])
    all_results["torch"] = {"installation": torch_result}
    
    # Test 3: Core dependencies
    print_header("CORE DEPENDENCIES")
    dep_results = test_core_dependencies()
    all_results["dependencies"] = dep_results
    
    for package, (passed, details) in dep_results.items():
        print_test_result(f"{package}", passed, details if not passed else "")
    
    # Test 4: MediaPipe functionality
    mp_result = test_mediapipe_functionality()
    print_test_result("MediaPipe Functionality", mp_result[0], mp_result[1])
    all_results["mediapipe"] = {"functionality": mp_result}
    
    # Test 5: GPU memory
    print_header("HARDWARE REQUIREMENTS")
    gpu_result = test_gpu_memory()
    print_test_result("GPU Memory", gpu_result[0], gpu_result[1])
    all_results["hardware"] = {"gpu_memory": gpu_result}
    
    # Test 6: ControlNet models
    print_header("CONTROLNET MODELS")
    model_results = test_controlnet_models()
    all_results["models"] = model_results
    
    for model_name, (passed, details) in model_results.items():
        print_test_result(f"{model_name}", passed, details if not passed else "")
    
    # Test 7: Kanibus nodes
    print_header("KANIBUS NODES")
    node_results = test_kanibus_nodes()
    all_results["nodes"] = node_results
    
    for node_name, (passed, details) in node_results.items():
        print_test_result(f"{node_name}", passed, details if not passed else "")
    
    return all_results

def print_summary(results: Dict[str, Dict[str, Tuple[bool, str]]]):
    """Print test summary"""
    print_header("TEST SUMMARY")
    
    total_tests = 0
    passed_tests = 0
    critical_failures = []
    
    for category, tests in results.items():
        for test_name, (passed, details) in tests.items():
            total_tests += 1
            if passed:
                passed_tests += 1
            else:
                # Identify critical failures
                if category in ["python", "torch"] or "control_" in test_name:
                    critical_failures.append(f"{category}.{test_name}")
    
    print(f"Overall: {passed_tests}/{total_tests} tests passed")
    
    if critical_failures:
        print(f"\n{Colors.RED}❌ Critical Issues Found:{Colors.END}")
        for failure in critical_failures:
            print(f"  • {failure}")
        print(f"\n{Colors.YELLOW}⚠️ Kanibus may not work properly until these issues are resolved{Colors.END}")
    else:
        print(f"\n{Colors.GREEN}✅ All critical tests passed! Kanibus should work properly.{Colors.END}")
    
    # Provide guidance
    missing_models = [name for name, (passed, _) in results.get("models", {}).items() if not passed]
    if missing_models:
        print(f"\n{Colors.BLUE}ℹ️ To download missing models, run:{Colors.END}")
        print(f"    python download_models.py")

def main():
    parser = argparse.ArgumentParser(description="Test Kanibus installation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fix-issues", action="store_true", help="Attempt to fix common issues")
    
    args = parser.parse_args()
    
    if args.fix_issues:
        fixes = fix_common_issues(args.verbose)
        print("\nFix Results:")
        for fix_name, success in fixes.items():
            print_test_result(fix_name, success)
    
    # Run all tests
    results = run_all_tests(args.verbose)
    
    # Print summary
    print_summary(results)
    
    # Determine exit code
    critical_tests = []
    for category, tests in results.items():
        for test_name, (passed, _) in tests.items():
            if category in ["python", "torch"] or "control_" in test_name:
                critical_tests.append(passed)
    
    if all(critical_tests):
        return 0  # Success
    else:
        return 1  # Critical failures

if __name__ == "__main__":
    sys.exit(main())