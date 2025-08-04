#!/usr/bin/env python3
"""
Script de instalação das dependências do Kanibus
Execute este script para instalar todas as dependências necessárias
"""

import subprocess
import sys
import os

def run_command(command, description=""):
    """Executa comando com tratamento de erro"""
    print(f"\n>>> {description or command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"OK: {description}")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERRO: {description}")
        print(f"Comando: {command}")
        if e.stderr:
            print(f"Erro: {e.stderr}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        return False

def main():
    print("=" * 60)
    print("INSTALADOR DE DEPENDENCIAS DO KANIBUS")
    print("=" * 60)
    
    # Lista de dependências essenciais (Atualizada para 2025)
    dependencies = [
        # Core ML/CV
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "torchaudio>=2.0.0",
        "mediapipe>=0.10.8",
        "opencv-python>=4.8.1.78",
        "opencv-contrib-python>=4.8.1.78",
        
        # GPU e Sistema
        "GPUtil>=1.4.0",
        "psutil>=5.9.0",
        
        # Computer Vision & ML
        "transformers>=4.35.0",
        "diffusers>=0.24.0",  # NOVO: Para suporte T2I-Adapter
        "ultralytics>=8.0.0",
        "timm>=0.9.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "scikit-image>=0.21.0",
        
        # Deep Learning Optimization
        "torchmetrics>=1.2.0",
        "albumentations>=1.3.0",
        "accelerate>=0.25.0",  # NOVO: Para otimização de modelos
        
        # Data Processing
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pillow>=10.0.0",
        "imageio>=2.31.0",
        "imageio-ffmpeg>=0.4.9",
        
        # Performance
        "numba>=0.58.0",
        
        # Utilities
        "tqdm>=4.66.0",
        "pyyaml>=6.0.1",
        "python-dotenv>=1.0.0",
    ]
    
    print(f"\nInstalando {len(dependencies)} dependencias essenciais...")
    
    # 1. Atualizar pip
    print("\n" + "="*50)
    print("1. ATUALIZANDO PIP")
    print("="*50)
    if not run_command(f'"{sys.executable}" -m pip install --upgrade pip', "Atualizando pip"):
        print("AVISO: Falha ao atualizar pip, continuando...")
    
    # 2. Instalar PyTorch primeiro (com CUDA se disponível)
    print("\n" + "="*50) 
    print("2. INSTALANDO PYTORCH")
    print("="*50)
    torch_command = f'"{sys.executable}" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121'
    if not run_command(torch_command, "Instalando PyTorch com CUDA"):
        # Fallback para versão CPU
        print("Tentando versão CPU do PyTorch...")
        torch_cpu_command = f'"{sys.executable}" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu'
        if not run_command(torch_cpu_command, "Instalando PyTorch CPU"):
            print("ERRO CRITICO: Falha ao instalar PyTorch")
            return False
    
    # 3. Instalar outras dependências
    print("\n" + "="*50)
    print("3. INSTALANDO OUTRAS DEPENDENCIAS")
    print("="*50)
    
    success_count = 0
    failed_packages = []
    
    for package in dependencies:
        if package.startswith("torch"):
            print(f"SKIP: {package} (ja instalado)")
            continue
            
        if run_command(f'"{sys.executable}" -m pip install "{package}"', f"Instalando {package}"):
            success_count += 1
        else:
            failed_packages.append(package)
    
    # 4. Relatório final
    print("\n" + "="*60)
    print("RELATORIO DE INSTALACAO")
    print("="*60)
    print(f"Pacotes instalados com sucesso: {success_count}")
    print(f"Pacotes com falha: {len(failed_packages)}")
    
    if failed_packages:
        print(f"\nPacotes que falharam:")
        for package in failed_packages:
            print(f"  - {package}")
        print(f"\nVoce pode tentar instalar manualmente:")
        for package in failed_packages:
            print(f'"{sys.executable}" -m pip install "{package}"')
    
    # 5. Teste rápido
    print(f"\n" + "="*50)
    print("5. TESTE RAPIDO")
    print("="*50)
    
    try:
        import torch
        print(f"OK: PyTorch {torch.__version__} instalado")
        print(f"CUDA disponivel: {torch.cuda.is_available()}")
    except ImportError:
        print("ERRO CRITICO: PyTorch nao foi instalado corretamente")
        return False
        
    try:
        import cv2
        print(f"OK: OpenCV {cv2.__version__} instalado")
    except ImportError:
        print("ERRO: OpenCV nao instalado")
        
    try:
        import mediapipe
        print(f"OK: MediaPipe instalado")
    except ImportError:
        print("ERRO: MediaPipe nao instalado")
    
    print(f"\n" + "="*60)
    if len(failed_packages) < 5:  # Se menos de 5 pacotes falharam
        print("INSTALACAO CONCLUIDA COM SUCESSO!")
        print("Agora você pode:")
        print("1. Reiniciar o ComfyUI")
        print("2. Executar: python download_models.py")
        print("3. Os nodes Kanibus devem aparecer na interface")
    else:
        print("INSTALACAO COM PROBLEMAS")
        print("Muitos pacotes falharam. Verifique sua conexao de internet")
        print("e tente novamente.")
    print("="*60)
    
    return len(failed_packages) < 5

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)