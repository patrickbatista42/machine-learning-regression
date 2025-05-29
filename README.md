# Projeto de Regressão para Machine Learning

Este projeto implementa um sistema de regressão usando machine learning para predição de preços de carros. Abaixo está a estrutura do projeto e a descrição de cada componente.

## Estrutura do Projeto

### Arquivos Principais
- `train_mod.csv` - Dataset original contendo os dados dos carros
- `README.md` - Este arquivo de documentação

### Pasta PreProcessamento/
Contém os scripts relacionados ao pré-processamento dos dados:
- `main.py` - Script principal que executa todo o pipeline de pré-processamento
- `preprocessamento.py` - Funções auxiliares para pré-processamento dos dados
- `preprocessamento_com_outlier.py` - Versão alternativa que mantém os outliers
- Arquivos gerados:
  - `ProcessedDatabase_SEM_outliers.csv` - Dataset processado sem outliers
  - `ProcessedDatabase_target_SEM_outliers.csv` - Variável target (preços) sem outliers

### Pasta Estudo_metodos/
Contém os scripts para implementação e avaliação dos modelos:
- `mainTest.py` - Script principal para testes dos modelos
- `modelos.py` - Implementação dos diferentes modelos de machine learning
- `otimizacao.py` - Funções para otimização de hiperparâmetros
- `feature_engineering.py` - Funções para engenharia de features
- `avaliacao.py` - Métricas e funções de avaliação dos modelos

### Arquivos Processados
Diferentes versões do dataset processado:
- `ProcessedDatabase_SEM_outliers.csv` - Dataset completo sem outliers
- `ProcessedDatabase_COM_outliers.csv` - Dataset completo com outliers
- `ProcessedDatabase_target_SEM_outliers.csv` - Apenas preços (target) sem outliers
- `ProcessedDatabase_target_COM_outliers.csv` - Apenas preços (target) com outliers
- `ProcessedDatabase_target.csv` - Versão base dos preços

## Pipeline de Pré-processamento

O pré-processamento dos dados inclui:
1. Limpeza de dados faltantes
2. Tratamento de outliers
3. Codificação de variáveis categóricas
4. Normalização dos dados
5. Análise de correlações
6. Redução de dimensionalidade (PCA)

## Modelos Implementados

Os modelos de machine learning incluem:
1. Regressão Linear
2. Random Forest
3. Gradient Boosting
4. Support Vector Regression (SVR)

## Como Usar

1. Execute o pré-processamento:
```bash
cd PreProcessamento
python main.py
```

2. Execute os modelos:
```bash
cd Estudo_metodos
python mainTest.py
```

## Requisitos
- Python 3.x
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib

## Resultados
Os resultados do processamento são salvos nos arquivos CSV correspondentes, permitindo análise posterior e comparação entre diferentes abordagens (com/sem outliers).
