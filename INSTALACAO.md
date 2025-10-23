# 📦 Guia de Instalação - Trabalho Final NLP

## Pré-requisitos

- **Python 3.8 ou superior**
- **pip** (gerenciador de pacotes Python)
- **Ambiente virtual** (recomendado)

---

## 🚀 Instalação Passo a Passo

### 1. Criar Ambiente Virtual (Recomendado)

#### Windows (PowerShell/CMD)
```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/Mac
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 2. Instalar Dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Tempo estimado**: 5-10 minutos (depende da conexão)

---

### 3. Instalar Modelo spaCy Português

O modelo `pt_core_news_sm` é necessário para lematização:

```bash
python -m spacy download pt_core_news_sm
```

**Tamanho**: ~15 MB

---

### 4. Download dos Recursos NLTK

Os recursos necessários serão baixados automaticamente ao executar o notebook, mas você pode baixá-los manualmente:

```python
import nltk
nltk.download('stopwords')
nltk.download('rslp')
nltk.download('punkt')
```

---

## ✅ Verificar Instalação

Execute o seguinte script para verificar se tudo está instalado corretamente:

```python
# verificar_instalacao.py

print("Verificando instalação das bibliotecas...")

try:
    import pandas as pd
    print("✓ pandas:", pd.__version__)
except ImportError:
    print("✗ pandas não instalado")

try:
    import numpy as np
    print("✓ numpy:", np.__version__)
except ImportError:
    print("✗ numpy não instalado")

try:
    import sklearn
    print("✓ scikit-learn:", sklearn.__version__)
except ImportError:
    print("✗ scikit-learn não instalado")

try:
    import nltk
    print("✓ nltk:", nltk.__version__)
except ImportError:
    print("✗ nltk não instalado")

try:
    import spacy
    print("✓ spacy:", spacy.__version__)
    try:
        nlp = spacy.load('pt_core_news_sm')
        print("✓ Modelo spaCy pt_core_news_sm instalado")
    except:
        print("✗ Modelo spaCy pt_core_news_sm NÃO instalado")
        print("  Execute: python -m spacy download pt_core_news_sm")
except ImportError:
    print("✗ spacy não instalado")

try:
    from sentence_transformers import SentenceTransformer
    print("✓ sentence-transformers instalado")
except ImportError:
    print("✗ sentence-transformers não instalado")

try:
    import matplotlib
    print("✓ matplotlib:", matplotlib.__version__)
except ImportError:
    print("✗ matplotlib não instalado")

try:
    import seaborn as sns
    print("✓ seaborn:", sns.__version__)
except ImportError:
    print("✗ seaborn não instalado")

try:
    from wordcloud import WordCloud
    print("✓ wordcloud instalado")
except ImportError:
    print("✗ wordcloud não instalado")

print("\n✅ Verificação concluída!")
```

---

## 📋 Lista de Pacotes Instalados

| Pacote | Versão Mínima | Descrição |
|--------|---------------|-----------|
| pandas | 1.5.0 | Manipulação de dados |
| numpy | 1.23.0 | Computação numérica |
| scikit-learn | 1.2.0 | Machine Learning |
| nltk | 3.8.0 | Processamento de texto |
| spacy | 3.5.0 | NLP avançado |
| sentence-transformers | 2.2.0 | Embeddings contextuais |
| torch | 2.0.0 | Deep Learning (backend) |
| matplotlib | 3.5.0 | Visualização |
| seaborn | 0.12.0 | Visualização estatística |
| wordcloud | 1.9.0 | Nuvens de palavras |
| joblib | 1.2.0 | Persistência de modelos |

---

## 🐛 Problemas Comuns

### Erro ao instalar torch

**Windows**: Instale a versão CPU:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Linux/Mac**: Geralmente instala sem problemas via pip.

### Erro ao baixar modelo spaCy

Se o comando `python -m spacy download pt_core_news_sm` falhar, baixe manualmente:

```bash
pip install https://github.com/explosion/spacy-models/releases/download/pt_core_news_sm-3.5.0/pt_core_news_sm-3.5.0-py3-none-any.whl
```

### Erro de memória ao carregar embeddings

O modelo `paraphrase-multilingual-MiniLM-L12-v2` requer ~500MB RAM. Se houver erro:
- Feche outros programas
- Use modelo menor: `paraphrase-multilingual-mpnet-base-v2`

### ImportError no Jupyter/Colab

Se estiver usando **Google Colab**, adicione no início do notebook:

```python
!pip install -r requirements.txt
!python -m spacy download pt_core_news_sm
```

---

## 💾 Requisitos de Sistema

### Mínimo
- **RAM**: 4 GB
- **Espaço em disco**: 2 GB
- **Processador**: Dual-core 2.0 GHz

### Recomendado
- **RAM**: 8 GB ou mais
- **Espaço em disco**: 5 GB
- **Processador**: Quad-core 2.5 GHz ou superior

---

## 🌐 Google Colab

Para executar no **Google Colab**, não é necessário instalar nada localmente. Apenas:

1. Faça upload do notebook
2. Execute a primeira célula que instala as dependências
3. O Colab já possui Python 3.10+ e pip

---

## 📞 Suporte

Se encontrar problemas na instalação:

1. Verifique a versão do Python: `python --version`
2. Atualize pip: `pip install --upgrade pip`
3. Limpe cache: `pip cache purge`
4. Reinstale em ambiente limpo (novo venv)

---

**Última atualização**: Outubro 2025

