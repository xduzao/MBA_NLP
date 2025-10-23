"""
Script de Verificação de Instalação
Trabalho Final NLP - QuantumFinance

Execute este script para verificar se todas as dependências
estão instaladas corretamente.

Uso:
    python verificar_instalacao.py
"""

import sys

def verificar_biblioteca(nome, nome_import=None):
    """Verifica se uma biblioteca está instalada"""
    if nome_import is None:
        nome_import = nome
    
    try:
        modulo = __import__(nome_import)
        versao = getattr(modulo, '__version__', 'versão desconhecida')
        print(f"✓ {nome}: {versao}")
        return True
    except ImportError:
        print(f"✗ {nome} NÃO instalado")
        return False

def verificar_spacy_modelo():
    """Verifica se o modelo spaCy português está instalado"""
    try:
        import spacy
        nlp = spacy.load('pt_core_news_sm')
        print("✓ Modelo spaCy pt_core_news_sm: instalado")
        return True
    except:
        print("✗ Modelo spaCy pt_core_news_sm: NÃO instalado")
        print("  → Execute: python -m spacy download pt_core_news_sm")
        return False

def verificar_nltk_recursos():
    """Verifica recursos NLTK"""
    try:
        import nltk
        recursos_necessarios = ['stopwords', 'rslp', 'punkt']
        faltando = []
        
        for recurso in recursos_necessarios:
            try:
                nltk.data.find(f'corpora/{recurso}' if recurso == 'stopwords' else f'tokenizers/{recurso}' if recurso == 'punkt' else f'stemmers/{recurso}')
            except LookupError:
                faltando.append(recurso)
        
        if not faltando:
            print(f"✓ Recursos NLTK: todos instalados")
            return True
        else:
            print(f"⚠ Recursos NLTK faltando: {', '.join(faltando)}")
            print(f"  → Serão baixados automaticamente ao executar o notebook")
            return True
    except:
        return True

def main():
    print("=" * 70)
    print("VERIFICAÇÃO DE INSTALAÇÃO - TRABALHO FINAL NLP")
    print("=" * 70)
    print()
    
    print(f"Python: {sys.version}")
    print()
    
    print("Verificando bibliotecas principais...")
    print("-" * 70)
    
    bibliotecas = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
        ('nltk', 'nltk'),
        ('spacy', 'spacy'),
        ('sentence-transformers', 'sentence_transformers'),
        ('torch', 'torch'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('wordcloud', 'wordcloud'),
        ('joblib', 'joblib'),
    ]
    
    todas_ok = True
    for nome, nome_import in bibliotecas:
        if not verificar_biblioteca(nome, nome_import):
            todas_ok = False
    
    print()
    print("Verificando componentes adicionais...")
    print("-" * 70)
    
    if not verificar_spacy_modelo():
        todas_ok = False
    
    verificar_nltk_recursos()
    
    print()
    print("=" * 70)
    
    if todas_ok:
        print("✅ TODAS AS DEPENDÊNCIAS ESTÃO INSTALADAS!")
        print()
        print("Você está pronto para executar o notebook:")
        print("  Trabalho_Final_NLP_QuantumFinance.ipynb")
    else:
        print("⚠ ALGUMAS DEPENDÊNCIAS ESTÃO FALTANDO")
        print()
        print("Execute os seguintes comandos:")
        print("  1. pip install -r requirements.txt")
        print("  2. python -m spacy download pt_core_news_sm")
    
    print("=" * 70)

if __name__ == "__main__":
    main()

