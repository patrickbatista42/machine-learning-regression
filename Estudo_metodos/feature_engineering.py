import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.impute import SimpleImputer

def tratar_nan(df):

    # Identificando colunas com NaN
    colunas_com_nan = df.columns[df.isna().any()].tolist()
    if colunas_com_nan:
        print(f"\nTratando valores NaN em {len(colunas_com_nan)} colunas:")
        for col in colunas_com_nan:
            nan_count = df[col].isna().sum()
            print(f"- {col}: {nan_count} valores NaN")
        
        # Usando SimpleImputer para preencher NaN
        imputer = SimpleImputer(strategy='median')
        df[colunas_com_nan] = imputer.fit_transform(df[colunas_com_nan])
    
    return df

def criar_interacoes_numericas(df, colunas_numericas):
    interacoes = []
    for i, col1 in enumerate(colunas_numericas):
        for col2 in colunas_numericas[i+1:]:
            nome = f"{col1}_x_{col2}"
            df[nome] = df[col1] * df[col2]
            interacoes.append(nome)
    return interacoes

def criar_features_polinomiais(df, colunas_numericas, grau=2):
    polinomiais = []
    for col in colunas_numericas:
        nome = f"{col}_quad"
        df[nome] = df[col] ** grau
        polinomiais.append(nome)
    return polinomiais

def criar_agregacoes_grupo(df, colunas_grupo, colunas_agg):
    agregacoes = []
    for col_grupo in colunas_grupo:
        for col_agg in colunas_agg:
            # Média do grupo
            nome_media = f"{col_grupo}_{col_agg}_media"
            df[nome_media] = df.groupby(col_grupo)[col_agg].transform('mean')
            agregacoes.append(nome_media)
            
            # Desvio padrão do grupo
            nome_std = f"{col_grupo}_{col_agg}_std"
            df[nome_std] = df.groupby(col_grupo)[col_agg].transform('std')
            agregacoes.append(nome_std)
    return agregacoes

def criar_features_categoricas_agregadas(df, coluna_grupo, coluna_valor):

    agregacoes = {}
    
    # Média do grupo
    nome_media = f"{coluna_grupo}_{coluna_valor}_media"
    agregacoes[nome_media] = df.groupby(coluna_grupo)[coluna_valor].transform('mean')
    
    # Desvio padrão do grupo
    nome_std = f"{coluna_grupo}_{coluna_valor}_std"
    agregacoes[nome_std] = df.groupby(coluna_grupo)[coluna_valor].transform('std')
    
    # Mediana do grupo
    nome_mediana = f"{coluna_grupo}_{coluna_valor}_mediana"
    agregacoes[nome_mediana] = df.groupby(coluna_grupo)[coluna_valor].transform('median')
    
    # Diferença da média do grupo
    df[nome_media] = agregacoes[nome_media]
    nome_diff = f"{coluna_grupo}_{coluna_valor}_diff_media"
    agregacoes[nome_diff] = df[coluna_valor] - agregacoes[nome_media]
    
    # Adicionando as agregações ao dataframe
    for nome, valores in agregacoes.items():
        df[nome] = valores
    
    return df, list(agregacoes.keys())

def aplicar_feature_engineering(df):
    # Cópia do DataFrame para não modificar o original
    df_eng = df.copy()
    
    # Identificar colunas numéricas e categóricas
    colunas_numericas = ['Km', 'Volume_motor', 'Cilindros', 'Portas', 'Rodas', 'Airbags']
    
    novas_features = []
    
    # 1. Interações numéricas
    print("Criando interações numéricas...")
    interacoes = criar_interacoes_numericas(df_eng, ['Km', 'Volume_motor'])
    novas_features.extend(interacoes)
    
    # 2. Features polinomiais
    print("Criando features polinomiais...")
    polinomiais = criar_features_polinomiais(df_eng, ['Km', 'Volume_motor'])
    novas_features.extend(polinomiais)
    
    # 3. Agregações por grupo
    print("Criando agregações por grupo...")
    agregacoes = criar_agregacoes_grupo(
        df_eng,
        colunas_grupo=['Fabricante', 'Modelo', 'Tipo_cambio'],
        colunas_agg=['Km', 'Volume_motor']
    )
    novas_features.extend(agregacoes)
    
    # 4. Tratamento de valores ausentes
    print("Tratando valores ausentes...")
    for col in df_eng.columns:
        if df_eng[col].isnull().any():
            if df_eng[col].dtype in ['int64', 'float64']:
                df_eng[col].fillna(df_eng[col].median(), inplace=True)
            else:
                df_eng[col].fillna(df_eng[col].mode()[0], inplace=True)
    
    # 5. Normalização das features numéricas
    print("Normalizando features numéricas...")
    scaler = StandardScaler()
    colunas_para_normalizar = colunas_numericas + novas_features
    df_eng[colunas_para_normalizar] = scaler.fit_transform(df_eng[colunas_para_normalizar])
    
    print(f"\nFeature engineering concluído.")
    print(f"Total de features originais: {len(df_eng.columns)}")
    print(f"Total de novas features: {len(novas_features)}")
    print("Exemplos de novas features:", novas_features[:5])
    
    return df_eng, novas_features 
