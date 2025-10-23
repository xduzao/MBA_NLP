# Trabalho Final NLP - QuantumFinance

## 📋 Descrição

Notebook completo desenvolvido para a disciplina de NLP (Processamento de Linguagem Natural) com foco na classificação automática de chamados de atendimento da QuantumFinance.

## 🎯 Objetivo

Criar um classificador de chamados aplicando técnicas de PLN que atinja **F1-Score ≥ 75%** no conjunto de teste.

## 📁 Arquivo Gerado

- **Trabalho_Final_NLP_QuantumFinance.ipynb**: Notebook principal com implementação completa

## 🏗️ Estrutura do Notebook

### 1. Setup Inicial e Carregamento dos Dados
- Importação de todas as bibliotecas necessárias
- Carregamento do dataset
- Separação estratificada treino/teste (75%/25%, random_state=42)

### 2. Análise Exploratória dos Dados (EDA)
**Com justificativas detalhadas**
- Verificações básicas (nulos, duplicados, textos vazios)
- Distribuição de classes (gráficos e percentuais)
- Análise de comprimento dos textos
- N-gramas mais frequentes (unigramas e bigramas)
- Nuvens de palavras por categoria

### 3. Pré-processamento de Texto
**Com justificativas sobre escolhas**
- Funções de limpeza (lowercase, pontuação, números)
- Stopwords pt-BR + customizadas
- Lematização com spaCy (foco em NOUN, VERB, ADJ)
- Comparação stemização vs lematização
- Transformador personalizado para Pipeline sklearn

### 4. Experimentos com Modelos Supervisionados

#### Experimento 1: TF-IDF + Regressão Logística
- GridSearchCV com 5-fold estratificado
- Hiperparâmetros: ngram_range, sublinear_tf, min_df, C
- Análise de features mais importantes por classe

#### Experimento 2: TF-IDF + Linear SVM
- GridSearchCV com 5-fold estratificado
- Hiperparâmetros: ngram_range, sublinear_tf, min_df, C, loss
- Comparação com Regressão Logística

#### Experimento 3: Sentence Embedding + Regressão Logística
- Transformer multilíngue: paraphrase-multilingual-MiniLM-L12-v2
- Extração de embeddings sem fine-tuning
- GridSearchCV para C da Regressão Logística

### 5. Comparação dos Modelos e Seleção do Campeão
- Tabela consolidada com métricas de todos os modelos
- Visualizações comparativas (F1-Score CV vs Teste, Tempo de treino)
- Seleção automática do melhor modelo

### 6. VALIDAÇÃO DO PROFESSOR
**Pipeline completo do modelo campeão reconstruído do zero**
- Carregamento dos dados
- Pré-processamento
- Construção do pipeline com melhores hiperparâmetros
- Treinamento
- Avaliação final (F1-Score, Accuracy, Classification Report)
- Matriz de confusão detalhada
- 10 exemplos de predições
- Persistência do modelo (joblib)
- Função de inferência para novos textos

### 7. Conclusões e Próximos Passos
- Resumo dos resultados
- Lições aprendidas
- Justificativa final da abordagem (tabela)
- Sugestões de melhorias futuras

## 🔧 Tecnologias e Bibliotecas

- **Python 3.x**
- **pandas, numpy**: Manipulação de dados
- **matplotlib, seaborn**: Visualizações
- **wordcloud**: Nuvens de palavras
- **nltk**: Stopwords, stemização (RSLP)
- **spacy**: Lematização, POS-tagging (pt_core_news_sm)
- **scikit-learn**: Vetorização (TF-IDF), modelos (LogisticRegression, LinearSVC), métricas, pipelines
- **sentence-transformers**: Embeddings contextuais
- **joblib**: Persistência de modelos

## 📊 Modelos Implementados

1. **TF-IDF + Regressão Logística**
   - Vetorização: TF-IDF com unigramas e bigramas
   - Classificador: Regressão Logística (OvR)
   
2. **TF-IDF + Linear SVM**
   - Vetorização: TF-IDF com unigramas e bigramas
   - Classificador: LinearSVC
   
3. **Sentence Embeddings + Regressão Logística**
   - Vetorização: Transformer multilíngue (embeddings densos)
   - Classificador: Regressão Logística

## 🎓 Justificativas Técnicas

Todas as decisões técnicas são justificadas com base:
- Nas características do dataset (textos curtos, domínio financeiro)
- Nas técnicas ensinadas na disciplina
- Nos resultados da análise exploratória
- Na literatura de NLP para classificação de textos

## ✅ Requisitos Atendidos

- ✅ Modelo classificador com técnicas de NLP
- ✅ Vetorização com n-gramas (TF-IDF 1-2 grams)
- ✅ Modelos supervisionados (Regressão Logística, SVM)
- ✅ Embeddings (Sentence Transformers)
- ✅ F1-Score > 75% (meta do enunciado)
- ✅ Split 75/25 estratificado (random_state=42)
- ✅ Pipeline completo do modelo campeão
- ✅ Justificativas para todas as decisões
- ✅ Reprodutibilidade garantida

## 🚀 Como Usar

1. **Executar o notebook completo**:
   - Abra o arquivo `Trabalho_Final_NLP_QuantumFinance.ipynb` no Jupyter/Colab
   - Execute todas as células sequencialmente
   
2. **Seção de Desenvolvimento**:
   - Contém EDA, pré-processamento e 3 experimentos
   - Todas as justificativas estão em células markdown
   
3. **Seção de Validação do Professor**:
   - Pipeline final do modelo campeão
   - Pronto para avaliação
   - Execução independente (não depende das seções anteriores)

## 📦 Dependências

### Instalação Rápida

```bash
# Instalar todas as dependências
pip install -r requirements.txt

# Download do modelo spaCy português
python -m spacy download pt_core_news_sm

# Verificar instalação
python verificar_instalacao.py
```

### Instalação Manual (alternativa)

```bash
pip install pandas numpy matplotlib seaborn wordcloud
pip install nltk spacy scikit-learn sentence-transformers joblib

# Download de recursos NLTK
python -c "import nltk; nltk.download('stopwords'); nltk.download('rslp'); nltk.download('punkt')"

# Download do modelo spaCy português
python -m spacy download pt_core_news_sm
```

### Arquivos de Suporte

- **`requirements.txt`**: Lista todas as dependências com versões
- **`INSTALACAO.md`**: Guia detalhado de instalação com troubleshooting
- **`verificar_instalacao.py`**: Script para verificar se tudo está instalado

## 📈 Métricas Principais

- **F1-Score (weighted)**: Métrica principal (requisito: ≥ 0.75)
- **Accuracy**: Métrica complementar
- **Classification Report**: Precision, Recall e F1 por classe
- **Matriz de Confusão**: Análise visual de erros

## 🎯 Diferenciais

1. **Justificativas Completas**: Todas as decisões são explicadas e fundamentadas
2. **Reprodutibilidade**: Pipeline completo do zero na seção de validação
3. **Comparação Robusta**: 3 abordagens diferentes com Grid Search
4. **Boas Práticas**: Pipeline sklearn, transformadores personalizados, persistência
5. **Análise Profunda**: EDA detalhada, features importantes, exemplos de predições

## 📝 Notas

- O notebook segue rigorosamente o template fornecido
- Todas as técnicas utilizadas foram vistas na disciplina
- O código está organizado, comentado e documentado
- Pronto para entrega e avaliação

---

**Desenvolvido para a disciplina de NLP - MBA**

