# 📋 WORKFLOW UPDATE NOTES (Janeiro 2025)

## ⚠️ **IMPORTANTE: Workflows Antigos Precisam de Atualização**

### **🔄 O que mudou:**

1. **Categoria Unificada**
   - **ANTES**: Nodes em subcategorias (Kanibus/Processing, Kanibus/Tracking, etc.)
   - **AGORA**: Todos os nodes estão sob categoria única "Kanibus"

2. **Novos Parâmetros**
   - `enable_t2i_adapter` - Suporte para T2I-Adapters (94% mais eficiente)
   - `wan_optimization` ou `wan_version` - Auto-detecção de WAN 2.1/2.2
   - Parâmetros extras de otimização em todos os nodes

3. **Compatibilidade Melhorada**
   - Suporte completo para T2I-Adapters
   - Detecção automática de versão WAN
   - Tratamento de erros robusto

### **🆕 Novos Workflows Criados (2025):**

1. **kanibus_basic_2025.json**
   - Workflow básico atualizado
   - Eye tracking + face masking
   - Categoria unificada

2. **kanibus_video_wan_2025.json**
   - Processamento de vídeo completo
   - Suporte WAN 2.1/2.2
   - Temporal smoothing
   - Multi-ControlNet

### **🔧 Como Atualizar Workflows Antigos:**

Se seus workflows não estão carregando corretamente:

1. **Abra o workflow no ComfyUI**
2. **Substitua nodes com erro** (ícone vermelho)
3. **Procure os nodes na categoria "Kanibus"** (não mais em subdirectórios)
4. **Reconecte as conexões**
5. **Adicione os novos parâmetros:**
   - `enable_t2i_adapter`: true
   - `wan_version`: "auto_detect"

### **📝 Exemplo de Atualização Manual:**

**ANTES (não funciona):**
```json
{
  "type": "NeuralPupilTracker",
  "properties": {"Node name for S&R": "NeuralPupilTracker"},
  "widgets_values": [1.0, 0.7, 0.2, 300.0, true, true, true, true]
}
```

**DEPOIS (funciona):**
```json
{
  "type": "NeuralPupilTracker",
  "properties": {"Node name for S&R": "NeuralPupilTracker"},
  "widgets_values": [1.0, 0.7, 0.2, 300.0, true, true, true, true, "auto_detect", true]
}
```

### **🚀 Recomendação:**

Use os **novos workflows 2025** como base para seus projetos. Eles já incluem:
- ✅ Categoria unificada
- ✅ Novos parâmetros
- ✅ Otimizações de performance
- ✅ Compatibilidade total

### **📁 Estrutura de Workflows:**

```
examples/
├── kanibus_basic_2025.json         # NOVO - Use este!
├── kanibus_video_wan_2025.json     # NOVO - Use este!
├── advanced_integrated_workflow.json # Antigo - precisa atualização
├── simple_eye_tracking.json         # Antigo - precisa atualização
├── wan21_basic_tracking.json       # Antigo - precisa atualização
└── wan22_advanced_full.json        # Antigo - precisa atualização
```

### **❓ Problemas Comuns:**

1. **"Node not found"** - Node está na categoria "Kanibus" agora
2. **"Missing input"** - Adicione os novos parâmetros
3. **"Invalid widget value"** - Valores precisam dos novos campos

### **💡 Dica:**

Se tiver muitos workflows para atualizar, comece criando novos baseados nos workflows 2025 - será mais rápido que corrigir os antigos!