#!/usr/bin/env python3
"""
Kanibus Installation Script - Automated setup for ComfyUI eye-tracking system
"""

import os
import sys
import subprocess
import platform
import pkg_resources
import urllib.request
import json
from pathlib import Path
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KanibusInstaller:
    """
    Automated installer for Kanibus eye-tracking system
    """
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.requirements_file = self.script_dir / "requirements.txt"
        self.system_info = self._get_system_info()
        self.installation_log = []
        
    def _get_system_info(self):
        """Get system information"""
        return {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "has_cuda": self._check_cuda_availability(),
            "has_mps": self._check_mps_availability(),
            "ram_gb": self._get_ram_info(),
        }
    
    def _check_cuda_availability(self):
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _check_mps_availability(self):
        """Check if MPS (Apple Silicon) is available"""
        try:
            import torch
            return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except ImportError:
            return False
    
    def _get_ram_info(self):
        """Get RAM information"""
        try:
            import psutil
            return round(psutil.virtual_memory().total / (1024**3))
        except ImportError:
            return 8  # Default assumption
    
    def _run_command(self, command, description=""):
        """Run command with logging"""
        logger.info(f"🔧 {description or command}")
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                check=True, 
                capture_output=True, 
                text=True
            )
            self.installation_log.append(f"✅ {description}: SUCCESS")
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            error_msg = f"❌ {description}: FAILED - {e.stderr or e.stdout}"
            logger.error(error_msg)
            self.installation_log.append(error_msg)
            return False, e.stderr or e.stdout
    
    def check_python_version(self):
        """Check Python version compatibility"""
        logger.info("🐍 Checking Python version...")
        
        version = sys.version_info
        if version.major != 3 or version.minor < 10:
            logger.error(f"❌ Python 3.10+ required, found {version.major}.{version.minor}")
            return False
        
        logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro} compatible")
        return True
    
    def install_dependencies(self):
        """Install Python dependencies"""
        logger.info("📦 Installing Python dependencies...")
        
        if not self.requirements_file.exists():
            logger.error(f"❌ Requirements file not found: {self.requirements_file}")
            return False
        
        # Upgrade pip first
        success, _ = self._run_command(
            f'"{sys.executable}" -m pip install --upgrade pip',
            "Upgrading pip"
        )
        if not success:
            logger.warning("⚠️ Failed to upgrade pip, continuing...")
        
        # Install requirements
        success, output = self._run_command(
            f'"{sys.executable}" -m pip install -r "{self.requirements_file}"',
            "Installing dependencies from requirements.txt"
        )
        
        return success
    
    def install_pytorch(self):
        """Install PyTorch with appropriate backend"""
        logger.info("🔥 Installing PyTorch...")
        
        if self.system_info["has_cuda"]:
            # Install CUDA version
            command = f'"{sys.executable}" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121'
            success, _ = self._run_command(command, "Installing PyTorch with CUDA 12.1")
        elif self.system_info["has_mps"]:
            # Install MPS version (Apple Silicon)
            command = f'"{sys.executable}" -m pip install torch torchvision torchaudio'
            success, _ = self._run_command(command, "Installing PyTorch with MPS support")
        else:
            # Install CPU version
            command = f'"{sys.executable}" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu'
            success, _ = self._run_command(command, "Installing PyTorch (CPU only)")
        
        return success
    
    def setup_directories(self):
        """Setup required directories"""
        logger.info("📁 Setting up directories...")
        
        directories = [
            self.script_dir / "cache",
            self.script_dir / "models",
            self.script_dir / "logs",
            self.script_dir / "examples" / "workflows",
            self.script_dir / "tests" / "fixtures",
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Created directory: {directory}")
            except Exception as e:
                logger.error(f"❌ Failed to create directory {directory}: {e}")
                return False
        
        return True
    
    def download_example_workflows(self):
        """Download example workflow files"""
        logger.info("📋 Creating example workflows...")
        
        workflows_dir = self.script_dir / "examples" / "workflows"
        
        # Create example workflows
        workflows = {
            "basic_eye_tracking.json": self._create_basic_workflow(),
            "wan21_video_processing.json": self._create_wan21_workflow(),
            "wan22_advanced_tracking.json": self._create_wan22_workflow(),
            "realtime_webcam.json": self._create_realtime_workflow()
        }
        
        for filename, workflow_data in workflows.items():
            try:
                workflow_file = workflows_dir / filename
                with open(workflow_file, 'w') as f:
                    json.dump(workflow_data, f, indent=2)
                logger.info(f"✅ Created workflow: {filename}")
            except Exception as e:
                logger.error(f"❌ Failed to create workflow {filename}: {e}")
                return False
        
        return True
    
    def _create_basic_workflow(self):
        """Create basic eye tracking workflow"""
        return {
            "nodes": {
                "1": {
                    "type": "VideoFrameLoader",
                    "inputs": {
                        "video_path": "input_video.mp4",
                        "quality": "high",
                        "color_space": "RGB"
                    }
                },
                "2": {
                    "type": "NeuralPupilTracker", 
                    "inputs": {
                        "image": ["1", 0],
                        "sensitivity": 1.0,
                        "smoothing": 0.7,
                        "enable_3d_gaze": True
                    }
                },
                "3": {
                    "type": "KanibusMaster",
                    "inputs": {
                        "input_source": "image",
                        "image": ["1", 0],
                        "pipeline_mode": "real_time",
                        "wan_version": "auto_detect",
                        "enable_eye_tracking": True
                    }
                }
            },
            "description": "Basic eye tracking workflow for single images or video frames"
        }
    
    def _create_wan21_workflow(self):
        """Create WAN 2.1 compatible workflow"""
        return {
            "nodes": {
                "1": {
                    "type": "VideoFrameLoader",
                    "inputs": {
                        "video_path": "input_video.mp4",
                        "resize_width": 854,
                        "resize_height": 480,
                        "target_fps": 24.0
                    }
                },
                "2": {
                    "type": "KanibusMaster",
                    "inputs": {
                        "input_source": "video",
                        "video_frames": ["1", 0],
                        "pipeline_mode": "batch",
                        "wan_version": "wan_2.1",
                        "target_fps": 24.0,
                        "enable_eye_tracking": True,
                        "enable_depth_estimation": True
                    }
                },
                "3": {
                    "type": "MultiControlNetApply",
                    "inputs": {
                        "eye_mask": ["2", 2],
                        "depth_map": ["2", 3],
                        "wan_version": "wan_2.1",
                        "eye_mask_weight": 1.2,
                        "depth_weight": 0.9
                    }
                }
            },
            "description": "WAN 2.1 optimized workflow (480p, 24fps)"
        }
    
    def _create_wan22_workflow(self):
        """Create WAN 2.2 compatible workflow"""
        return {
            "nodes": {
                "1": {
                    "type": "VideoFrameLoader",
                    "inputs": {
                        "video_path": "input_video.mp4",
                        "resize_width": 1280,
                        "resize_height": 720,
                        "target_fps": 30.0
                    }
                },
                "2": {
                    "type": "KanibusMaster",
                    "inputs": {
                        "input_source": "video",
                        "video_frames": ["1", 0],
                        "pipeline_mode": "streaming",
                        "wan_version": "wan_2.2",
                        "target_fps": 30.0,
                        "enable_eye_tracking": True,
                        "enable_face_tracking": True,
                        "enable_depth_estimation": True,
                        "enable_emotion_analysis": True
                    }
                },
                "3": {
                    "type": "MultiControlNetApply",
                    "inputs": {
                        "eye_mask": ["2", 2],
                        "depth_map": ["2", 3],
                        "normal_map": ["2", 4], 
                        "wan_version": "wan_2.2",
                        "eye_mask_weight": 1.3,
                        "depth_weight": 1.0,
                        "normal_weight": 0.7
                    }
                }
            },
            "description": "WAN 2.2 advanced workflow (720p, 30fps) with full feature set"
        }
    
    def _create_realtime_workflow(self):
        """Create real-time webcam workflow"""
        return {
            "nodes": {
                "1": {
                    "type": "KanibusMaster",
                    "inputs": {
                        "input_source": "webcam",
                        "pipeline_mode": "real_time",
                        "target_fps": 30.0,
                        "enable_eye_tracking": True,
                        "enable_face_tracking": True,
                        "tracking_quality": "medium"
                    }
                },
                "2": {
                    "type": "TemporalSmoother",
                    "inputs": {
                        "current_frame": ["1", 1],
                        "smoothing_strength": 0.8,
                        "adaptive_smoothing": True
                    }
                }
            },
            "description": "Real-time webcam processing with temporal smoothing"
        }
    
    def create_test_configuration(self):
        """Create test configuration"""
        logger.info("🧪 Creating test configuration...")
        
        config = {
            "system_info": self.system_info,
            "installation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "performance_targets": {
                "eye_tracking_fps": 60 if self.system_info["has_cuda"] else 30,
                "full_pipeline_fps": 24 if self.system_info["has_cuda"] else 12,
                "memory_usage_limit_gb": min(self.system_info["ram_gb"] * 0.8, 16)
            },
            "feature_compatibility": {
                "gpu_acceleration": self.system_info["has_cuda"] or self.system_info["has_mps"],
                "real_time_processing": self.system_info["ram_gb"] >= 8,
                "4k_processing": self.system_info["has_cuda"] and self.system_info["ram_gb"] >= 16,
                "batch_processing": True
            }
        }
        
        config_file = self.script_dir / "config.json"
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"✅ Created configuration: {config_file}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create configuration: {e}")
            return False
    
    def run_post_install_tests(self):
        """Run basic tests after installation"""
        logger.info("🧪 Running post-installation tests...")
        
        tests_passed = 0
        total_tests = 0
        
        # Test 1: Import core modules
        total_tests += 1
        try:
            import torch
            import cv2
            import mediapipe
            import numpy as np
            logger.info("✅ Core dependencies import successfully")
            tests_passed += 1
        except ImportError as e:
            logger.error(f"❌ Failed to import core dependencies: {e}")
        
        # Test 2: PyTorch backend
        total_tests += 1
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            tensor = torch.randn(100, 100).to(device)
            result = torch.matmul(tensor, tensor)
            logger.info(f"✅ PyTorch working on {device}")
            tests_passed += 1
        except Exception as e:
            logger.error(f"❌ PyTorch test failed: {e}")
        
        # Test 3: MediaPipe
        total_tests += 1
        try:
            import mediapipe as mp
            face_mesh = mp.solutions.face_mesh.FaceMesh()
            logger.info("✅ MediaPipe initialized successfully")
            tests_passed += 1
        except Exception as e:
            logger.error(f"❌ MediaPipe test failed: {e}")
        
        # Test 4: OpenCV
        total_tests += 1
        try:
            import cv2
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            logger.info("✅ OpenCV working correctly")
            tests_passed += 1
        except Exception as e:
            logger.error(f"❌ OpenCV test failed: {e}")
        
        success_rate = tests_passed / total_tests * 100
        logger.info(f"🎯 Post-installation tests: {tests_passed}/{total_tests} passed ({success_rate:.1f}%)")
        
        return success_rate >= 75  # Require 75% success rate
    
    def print_installation_summary(self, success):
        """Print installation summary"""
        print("\\n" + "="*60)
        print("🐝 KANIBUS INSTALLATION SUMMARY")
        print("="*60)
        
        print(f"Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
        print(f"Python: {self.system_info['python_version']}")
        print(f"Platform: {self.system_info['platform']} ({self.system_info['architecture']})")
        print(f"GPU Support: {'CUDA' if self.system_info['has_cuda'] else 'MPS' if self.system_info['has_mps'] else 'CPU Only'}")
        print(f"RAM: {self.system_info['ram_gb']} GB")
        
        print("\\n📋 Installation Log:")
        for log_entry in self.installation_log[-10:]:  # Show last 10 entries
            print(f"  {log_entry}")
        
        if success:
            print("\\n🚀 NEXT STEPS:")
            print("1. Restart ComfyUI")
            print("2. Look for 'Kanibus' category in node menu")
            print("3. Try example workflows in examples/workflows/")
            print("4. Check config.json for system optimization settings")
            print("\\n📖 Documentation: ./docs/README.md")
        else:
            print("\\n❌ TROUBLESHOOTING:")
            print("1. Check Python version (3.10+ required)")
            print("2. Ensure sufficient disk space (>5GB)")
            print("3. Check internet connection for downloads")
            print("4. See logs above for specific errors")
        
        print("="*60)
    
    def install(self):
        """Run complete installation"""
        logger.info("🐝 Starting Kanibus installation...")
        
        # Check prerequisites
        if not self.check_python_version():
            return False
        
        # Install components
        steps = [
            ("setup_directories", "Setting up directories"),
            ("install_pytorch", "Installing PyTorch"),
            ("install_dependencies", "Installing dependencies"), 
            ("download_example_workflows", "Creating example workflows"),
            ("create_test_configuration", "Creating configuration"),
            ("run_post_install_tests", "Running tests")
        ]
        
        for step_method, step_description in steps:
            logger.info(f"📍 {step_description}...")
            method = getattr(self, step_method)
            if not method():
                logger.error(f"❌ Failed: {step_description}")
                self.print_installation_summary(False)
                return False
        
        logger.info("✅ Installation completed successfully!")
        self.print_installation_summary(True)
        return True

def main():
    """Main installation function"""
    print("🐝 Kanibus Eye-Tracking ControlNet System")
    print("   Advanced ComfyUI Integration Installer")
    print("   Version 1.0.0\\n")
    
    installer = KanibusInstaller()
    success = installer.install()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()