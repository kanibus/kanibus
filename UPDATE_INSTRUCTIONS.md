# 🔄 INSTRUÇÕES DE ATUALIZAÇÃO DO KANIBUS

## 🚀 **ATUALIZAÇÃO RÁPIDA (Para quem já tem instalado)**

Execute estes comandos no Command Prompt como Administrador:

### **1. Backup (Opcional mas Recomendado)**
```cmd
cd /d I:\ComfyUI_windows_portable\ComfyUI\custom_nodes
rename Kanibus Kanibus_backup
```

### **2. Clonar Versão Atualizada do GitHub**
```cmd
git clone https://github.com/kanibus/kanibus.git Kanibus
```

### **3. Instalar/Atualizar Dependências**
```cmd
cd Kanibus
I:\ComfyUI_windows_portable\python_embeded\python.exe -m pip install --upgrade -r requirements.txt
```

### **4. Baixar Modelos T2I-Adapter (Se ainda não tiver)**
```cmd
I:\ComfyUI_windows_portable\python_embeded\python.exe download_models.py
```

### **5. Reiniciar ComfyUI**

---

## 📦 **INSTALAÇÃO COMPLETA (Primeira vez)**

### **1. Clonar do GitHub**
```cmd
cd /d I:\ComfyUI_windows_portable\ComfyUI\custom_nodes
git clone https://github.com/kanibus/kanibus.git Kanibus
cd Kanibus
```

### **2. Instalar Dependências**
```cmd
I:\ComfyUI_windows_portable\python_embeded\python.exe install_dependencies.py
```

### **3. Baixar Modelos**
```cmd
I:\ComfyUI_windows_portable\python_embeded\python.exe download_models.py
```

### **4. Verificar Instalação**
```cmd
I:\ComfyUI_windows_portable\python_embeded\python.exe test_installation.py
```

### **5. Reiniciar ComfyUI**

---

## 🆕 **O QUE HÁ DE NOVO (Janeiro 2025)**

### **✅ Todas as Atualizações Implementadas:**

1. **Categoria Unificada**
   - Todos os nodes agora aparecem sob "Kanibus" no menu
   - Não há mais subcategorias confusas

2. **Suporte T2I-Adapter**
   - 94% mais eficiente que ControlNet
   - Modelos de apenas 158MB cada
   - Download automático via script

3. **Compatibilidade WAN 2.1/2.2**
   - Detecção automática de versão
   - Otimizações específicas por versão
   - Parâmetros ajustados automaticamente

4. **Melhorias de Performance**
   - Uso otimizado de GPU
   - Cache inteligente
   - Processamento em batch

5. **14 Nodes Atualizados**
   - KanibusMaster
   - VideoFrameLoader
   - NeuralPupilTracker
   - AdvancedTrackingPro
   - SmartFacialMasking
   - AIDepthControl
   - NormalMapGenerator
   - LandmarkPro468
   - EmotionAnalyzer
   - HandTracking
   - BodyPoseEstimator
   - ObjectSegmentation
   - TemporalSmoother
   - MultiControlNetApply

---

## 🔧 **SOLUÇÃO DE PROBLEMAS**

### **Nodes não aparecem no ComfyUI:**
```cmd
# Reinstalar dependências
I:\ComfyUI_windows_portable\python_embeded\python.exe -m pip install --upgrade -r requirements.txt

# Verificar instalação
I:\ComfyUI_windows_portable\python_embeded\python.exe debug_imports.py
```

### **Erro de importação:**
```cmd
# Instalar PyTorch com CUDA
I:\ComfyUI_windows_portable\python_embeded\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### **Modelos não encontrados:**
```cmd
# Baixar modelos T2I-Adapter
I:\ComfyUI_windows_portable\python_embeded\python.exe download_models.py

# Ou baixar TODOS os modelos
I:\ComfyUI_windows_portable\python_embeded\python.exe download_models.py --all
```

---

## 📋 **VERIFICAÇÃO RÁPIDA**

Após atualizar, verifique:

1. ✅ Nodes aparecem sob categoria "Kanibus" no menu
2. ✅ Nenhum erro ao iniciar ComfyUI
3. ✅ Workflows carregam sem nodes faltando
4. ✅ GPU está sendo utilizada (se disponível)

---

## 🆘 **SUPORTE**

- **Issues**: https://github.com/kanibus/kanibus/issues
- **Discussões**: https://github.com/kanibus/kanibus/discussions
- **Email**: staffytech@proton.me

---

## 🔄 **HISTÓRICO DE ATUALIZAÇÕES**

- **v1.0.0 (Jan 2025)**: Lançamento completo com T2I-Adapters e WAN 2.1/2.2
- **Restoration Points**: 6 pontos de restauração para rollback seguro