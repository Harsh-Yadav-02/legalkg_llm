## 🧠 Tech Stack Used

<p align="left">
  <img src="https://skillicons.dev/icons?i=python,linux,git" height="35"/>
  <img src="https://upload.wikimedia.org/wikipedia/commons/2/21/Nvidia_logo.svg" height="30"/>
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30"/>
  <img src="https://ollama.com/public/ollama.png" height="30"/>
  <img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/serverfault.svg" height="30"/>
</p>

- **Programming Language**: Python  
- **LLMs**: DeepSeek (via Ollama)  
- **Model Hosting**: Hugging Face  
- **Compute**: NVIDIA A100 GPU  
- **OS**: Linux  
- **Execution Environment**: Dedicated Server  

## 🚀 Execution Environment

- 🐧 Linux-based server  
- ⚡ NVIDIA A100 (80GB VRAM)  
- 🧠 Ollama for local LLM inference  
- 🤗 Hugging Face for model access & experimentation  
- 🐍 Python-based pipeline

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![NVIDIA](https://img.shields.io/badge/NVIDIA-A100-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge)

```
.vscode/
    └── settings.json

datasets/
    ├── judgements/
    │   ├── 1953_24.txt
    │   ├── ...
    │   └── 2018_C_7.txt
    ├── abbreviations.json
    ├── contraction_map.txt
    ├── ...
    └── relations.txt

output/
    └── drive_link.md

prompts/
    ├── etype_postprocessing.txt
    ├── ...
    └── relations_definition.txt

scripts/
    ├── filkgc/
    │   └── entities_and_relations.py
    └── filox/
        ├── entity_types/
        │   ├── entity_relation_extraction.py
        │   ├── ...
        │   └── preprocessing.py
        ├── relations/
        │   ├── expansion_relations.py
        │   ├── ...
        │   └── relations_definition.py
        └── dataset_analysis.py

.gitignore
output.log
requirements.txt
```
