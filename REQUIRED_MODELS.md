# 📦 MODELOS NECESSÁRIOS PARA KANIBUS

## 🎯 **MODELOS OBRIGATÓRIOS**

Para o Kanibus funcionar completamente, você **DEVE** baixar estes 4 modelos ControlNet:

---

## 🎛️ **CONTROLNET MODELS (OBRIGATÓRIOS)**

### **📁 Localização no ComfyUI:**
```
ComfyUI/models/controlnet/
```

### **📥 DOWNLOADS OBRIGATÓRIOS:**

#### **1. 🎨 ControlNet Scribble (Para Eye Masks)**
- **Arquivo**: `control_v11p_sd15_scribble.pth`
- **Tamanho**: ~1.4GB
- **Uso**: Controle por máscaras de olhos
- **Download**: https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble.pth
- **Salvar em**: `ComfyUI/models/controlnet/control_v11p_sd15_scribble.pth`

#### **2. 🌊 ControlNet Depth (Para Depth Maps)**
- **Arquivo**: `control_v11f1p_sd15_depth.pth`
- **Tamanho**: ~1.4GB
- **Uso**: Controle por mapas de profundidade
- **Download**: https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth
- **Salvar em**: `ComfyUI/models/controlnet/control_v11f1p_sd15_depth.pth`

#### **3. 🗺️ ControlNet Normal (Para Normal Maps)**
- **Arquivo**: `control_v11p_sd15_normalbae.pth`
- **Tamanho**: ~1.4GB
- **Uso**: Controle por mapas normais de superfície
- **Download**: https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_normalbae.pth
- **Salvar em**: `ComfyUI/models/controlnet/control_v11p_sd15_normalbae.pth`

#### **4. 🏃 ControlNet OpenPose (Para Pose Detection)**
- **Arquivo**: `control_v11p_sd15_openpose.pth`
- **Tamanho**: ~1.4GB
- **Uso**: Controle por detecção de pose corporal
- **Download**: https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth
- **Salvar em**: `ComfyUI/models/controlnet/control_v11p_sd15_openpose.pth`

**📊 Total de espaço necessário**: ~5.6GB

---

## 🤖 **DOWNLOAD AUTOMÁTICO**

Use este script para baixar todos os modelos automaticamente:

```bash
# Windows (PowerShell)
.\download_models.ps1

# Linux/Mac
./download_models.sh

# Python (qualquer sistema)
python download_models.py
```

---

## ❌ **MODELOS QUE NÃO PRECISA BAIXAR**

Estes modelos são baixados **automaticamente** pelas bibliotecas:

### **👁️ Eye Tracking Models**
- ✅ **MediaPipe Face Mesh** (~6MB) - Auto-download
- ✅ **MediaPipe Iris** (~2MB) - Auto-download

### **🧠 AI Models**
- ✅ **DPT Depth Models** (~1.3GB) - Auto-download via Hugging Face
- ✅ **MiDaS Models** (~400MB) - Auto-download via Hugging Face
- ✅ **SAM Models** (~2.4GB) - Auto-download quando necessário
- ✅ **YOLO Models** (~6MB) - Auto-download via Ultralytics
- ✅ **Emotion Models** - Baseado em MediaPipe (auto-download)

---

## ✅ **VERIFICAÇÃO PÓS-INSTALAÇÃO**

Após baixar os modelos, verifique se estes arquivos existem:

```bash
ComfyUI/models/controlnet/
├── control_v11p_sd15_scribble.pth      ✅ (1.4GB)
├── control_v11f1p_sd15_depth.pth       ✅ (1.4GB)
├── control_v11p_sd15_normalbae.pth     ✅ (1.4GB)
└── control_v11p_sd15_openpose.pth      ✅ (1.4GB)
```

### **🧪 TESTE RÁPIDO**

Execute este comando para verificar se tudo está funcionando:

```bash
# No diretório do Kanibus
python test_installation.py
```

Se todos os testes passarem, você verá:
```
✅ Kanibus instalado corretamente
✅ Todos os 4 modelos ControlNet encontrados
✅ MediaPipe funcionando
✅ PyTorch com GPU detectado
✅ Sistema pronto para uso!
```

---

## 🚨 **PROBLEMAS COMUNS**

### **❌ Modelo não encontrado**
```
Error: ControlNet model not found: control_v11p_sd15_scribble.pth
```
**Solução**: Verifique se o arquivo está em `ComfyUI/models/controlnet/`

### **❌ Arquivo corrompido**
```
Error: Unable to load ControlNet model
```
**Solução**: Re-baixe o modelo - arquivo pode estar corrompido

### **❌ Espaço insuficiente**
```
Error: No space left on device
```
**Solução**: Libere pelo menos 6GB de espaço em disco

---

## 📞 **SUPORTE**

Se tiver problemas com os downloads:

- **Issues**: [GitHub Issues](https://github.com/kanibus/kanibus/issues)
- **Discussions**: [GitHub Discussions](https://github.com/kanibus/kanibus/discussions)
- **Email**: staffytech@proton.me

---

## 🔗 **LINKS ALTERNATIVOS**

Se os links do Hugging Face estiverem lentos, tente estes mirrors:

### **Mirror 1 - Civitai**
- Scribble: https://civitai.com/models/9251/controlnet-v11-sd15-scribble
- Depth: https://civitai.com/models/9251/controlnet-v11-sd15-depth
- Normal: https://civitai.com/models/9251/controlnet-v11-sd15-normalbae
- OpenPose: https://civitai.com/models/9251/controlnet-v11-sd15-openpose

### **Mirror 2 - ComfyUI Manager**
Se você tem o ComfyUI Manager instalado:
1. Abra ComfyUI Manager
2. Vá para "Model Manager"
3. Procure por "ControlNet v1.1"
4. Instale os 4 modelos listados acima

---

**⚠️ IMPORTANTE**: O Kanibus **NÃO FUNCIONARÁ** sem estes 4 modelos ControlNet. Certifique-se de baixá-los antes de usar!