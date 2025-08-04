# 📦 MODELOS NECESSÁRIOS PARA KANIBUS - WAN COMPATIBLE

## 🚀 **MODELOS WAN-OPTIMIZED (2025)**

Para máxima compatibilidade com WAN 2.1/2.2 e performance moderna, use os modelos atualizados:

---

## 🎛️ **T2I-ADAPTERS (RECOMENDADO - 94% MAIS EFICIENTE)**

### **✨ VANTAGENS DOS T2I-ADAPTERS:**
- 🚀 **94% menor**: 158MB vs 2.5GB por modelo
- ⚡ **93.69% menos parâmetros** que ControlNet
- 🎯 **Velocidade**: Impacto quase zero na geração
- 🎬 **Otimizado para vídeo**: Melhor consistência temporal
- 🔧 **WAN Compatible**: Funciona nativamente com WAN 2.1/2.2

### **📁 Localização no ComfyUI:**
```
ComfyUI/models/t2i_adapter/    # Para T2I-Adapters
ComfyUI/models/controlnet/     # Para modelos legados (backup)
```

### **📥 DOWNLOADS PRIMÁRIOS (T2I-ADAPTERS):**

#### **1. 🎨 T2I-Adapter Sketch (Para Eye Masks)**
- **Arquivo**: `t2iadapter_sketch_sd14v1.pth`
- **Tamanho**: ~158MB (vs 1.4GB ControlNet)
- **Uso**: Controle por máscaras de olhos e sketches
- **Download**: https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_sketch_sd14v1.pth
- **Salvar em**: `ComfyUI/models/t2i_adapter/t2iadapter_sketch_sd14v1.pth`

#### **2. 🌊 T2I-Adapter Depth (Para Depth Maps)**
- **Arquivo**: `t2iadapter_depth_sd14v1.pth`
- **Tamanho**: ~158MB (vs 1.4GB ControlNet)
- **Uso**: Controle por mapas de profundidade
- **Download**: https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_depth_sd14v1.pth
- **Salvar em**: `ComfyUI/models/t2i_adapter/t2iadapter_depth_sd14v1.pth`

#### **3. 🗺️ T2I-Adapter Canny (Para Edge Detection)**
- **Arquivo**: `t2iadapter_canny_sd14v1.pth`
- **Tamanho**: ~158MB (vs 1.4GB ControlNet)
- **Uso**: Controle por detecção de bordas
- **Download**: https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_canny_sd14v1.pth
- **Salvar em**: `ComfyUI/models/t2i_adapter/t2iadapter_canny_sd14v1.pth`

#### **4. 🏃 T2I-Adapter OpenPose (Para Pose Detection)**
- **Arquivo**: `t2iadapter_openpose_sd14v1.pth`
- **Tamanho**: ~158MB (vs 1.4GB ControlNet)
- **Uso**: Controle por detecção de pose corporal
- **Download**: https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_openpose_sd14v1.pth
- **Salvar em**: `ComfyUI/models/t2i_adapter/t2iadapter_openpose_sd14v1.pth`

**📊 Total T2I-Adapters**: ~632MB (vs 5.6GB ControlNet)

---

## 🎬 **MODELOS ESPECÍFICOS PARA VÍDEO**

### **📥 STABLE VIDEO DIFFUSION (SVD) ADAPTERS:**

#### **5. 🎥 SVD ControlNet for Video**
- **Arquivo**: `svd_controlnet.safetensors`
- **Tamanho**: ~2.1GB
- **Uso**: Controle temporal para vídeos WAN
- **Download**: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_controlnet.safetensors
- **Salvar em**: `ComfyUI/models/controlnet/svd_controlnet.safetensors`

#### **6. 🌊 I2V-Adapter (Image-to-Video)**
- **Arquivo**: `i2v_adapter.safetensors`
- **Tamanho**: ~850MB
- **Uso**: Conversão imagem para vídeo com controle
- **Download**: https://huggingface.co/TencentARC/I2V-Adapter/resolve/main/i2v_adapter.safetensors
- **Salvar em**: `ComfyUI/models/controlnet/i2v_adapter.safetensors`

---

## 🔄 **MODELOS LEGADOS (BACKUP COMPATIBILITY)**

### **📥 DOWNLOADS LEGADOS (Para compatibilidade com workflows antigos):**

#### **1. 🎨 ControlNet Scribble (Legacy)**
- **Arquivo**: `control_v11p_sd15_scribble.pth`
- **Tamanho**: ~1.4GB
- **Status**: LEGADO - Use T2I-Adapter
- **Download**: https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble.pth
- **Salvar em**: `ComfyUI/models/controlnet/control_v11p_sd15_scribble.pth`

#### **2. 🌊 ControlNet Depth (Legacy)**
- **Arquivo**: `control_v11f1p_sd15_depth.pth`
- **Tamanho**: ~1.4GB
- **Status**: LEGADO - Use T2I-Adapter
- **Download**: https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth
- **Salvar em**: `ComfyUI/models/controlnet/control_v11f1p_sd15_depth.pth`

#### **3. 🗺️ ControlNet Normal (Legacy)**
- **Arquivo**: `control_v11p_sd15_normalbae.pth`
- **Tamanho**: ~1.4GB
- **Status**: LEGADO - Use T2I-Adapter Canny
- **Download**: https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_normalbae.pth
- **Salvar em**: `ComfyUI/models/controlnet/control_v11p_sd15_normalbae.pth`

#### **4. 🏃 ControlNet OpenPose (Legacy)**
- **Arquivo**: `control_v11p_sd15_openpose.pth`
- **Tamanho**: ~1.4GB
- **Status**: LEGADO - Use T2I-Adapter
- **Download**: https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth
- **Salvar em**: `ComfyUI/models/controlnet/control_v11p_sd15_openpose.pth`

**📊 Total Legacy**: ~5.6GB

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