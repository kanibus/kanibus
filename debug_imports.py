#!/usr/bin/env python3
"""
Debug script para verificar problemas de importação do Kanibus
"""

import sys
import os
import traceback

# Adicionar diretório do Kanibus ao path
kanibus_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, kanibus_dir)

print("DEBUG: Kanibus Import Test")
print("=" * 50)

# Teste 1: Importar módulos core
print("\n1. Testando imports do core system...")
try:
    from src.neural_engine import NeuralEngine, ProcessingConfig, ProcessingMode
    print("OK: neural_engine importado com sucesso")
except Exception as e:
    print(f"ERRO: Erro ao importar neural_engine: {e}")
    print(traceback.format_exc())

try:
    from src.gpu_optimizer import GPUOptimizer
    print("OK: gpu_optimizer importado com sucesso")
except Exception as e:
    print(f"ERRO: Erro ao importar gpu_optimizer: {e}")
    print(traceback.format_exc())

try:
    from src.cache_manager import CacheManager
    print("OK: cache_manager importado com sucesso")
except Exception as e:
    print(f"ERRO: Erro ao importar cache_manager: {e}")
    print(traceback.format_exc())

# Teste 2: Importar nodes individuais
print("\n2. Testando imports dos nodes...")
nodes_to_test = [
    "kanibus_master",
    "video_frame_loader", 
    "neural_pupil_tracker",
    "advanced_tracking_pro",
    "smart_facial_masking",
    "ai_depth_control",
    "normal_map_generator",
    "landmark_pro_468",
    "emotion_analyzer",
    "hand_tracking",
    "body_pose_estimator",
    "object_segmentation",
    "temporal_smoother",
    "multi_controlnet_apply"
]

successful_imports = []
failed_imports = []

for node_name in nodes_to_test:
    try:
        module = __import__(f"nodes.{node_name}", fromlist=[node_name])
        print(f"OK: {node_name} importado com sucesso")
        successful_imports.append(node_name)
    except Exception as e:
        print(f"ERRO: Erro ao importar {node_name}: {e}")
        failed_imports.append((node_name, str(e)))

# Teste 3: Importar o __init__.py principal
print("\n3. Testando import do sistema completo...")
try:
    from nodes import *
    print("OK: nodes.__init__ importado com sucesso")
except Exception as e:
    print(f"ERRO: Erro ao importar nodes.__init__: {e}")
    print(traceback.format_exc())

# Teste 4: Testar NODE_CLASS_MAPPINGS
print("\n4. Testando NODE_CLASS_MAPPINGS...")
try:
    from nodes import NODE_CLASS_MAPPINGS
    print(f"OK: NODE_CLASS_MAPPINGS encontrado com {len(NODE_CLASS_MAPPINGS)} nodes")
    for node_name in NODE_CLASS_MAPPINGS:
        print(f"  - {node_name}")
except Exception as e:
    print(f"ERRO: Erro ao importar NODE_CLASS_MAPPINGS: {e}")

# Resumo
print(f"\n" + "=" * 50)
print("RESUMO:")
print(f"OK: Imports bem-sucedidos: {len(successful_imports)}")
print(f"ERRO: Imports com falha: {len(failed_imports)}")

if failed_imports:
    print(f"\nProblemas encontrados:")
    for node_name, error in failed_imports:
        print(f"  - {node_name}: {error}")

print(f"\nPara corrigir, verifique:")
print(f"  1. Dependencias instaladas: pip install -r requirements.txt")
print(f"  2. Estrutura de arquivos completa")
print(f"  3. Sintaxe dos arquivos Python")