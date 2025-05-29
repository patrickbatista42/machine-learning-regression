import numpy as np
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from joblib import Parallel, delayed

def otimizar_random_forest(X_train, y_train, cv=10, n_iter=50):
    """
    Otimiza os hiperparâmetros do Random Forest usando RandomizedSearchCV.
    """
    # Convertendo para array se necessário
    if hasattr(y_train, 'values'):
        y_train = y_train.values.ravel()
    
    # Definir o espaço de busca dos hiperparâmetros
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None] + list(range(10, 110, 20)),
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt'],
        'bootstrap': [True, False]
    }
    
    # Criar o modelo base
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Configurar e executar a busca aleatória
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    # Realizar a busca
    random_search.fit(X_train, y_train)
    
    # Imprimir resultados
    print("\nMelhores hiperparâmetros encontrados:")
    for param, value in random_search.best_params_.items():
        print(f"{param}: {value}")
    print(f"\nMelhor score (R²): {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_params_, random_search.best_score_ 

class SVROptimization(Problem):
    """
    Classe para otimização dos hiperparâmetros do SVR usando NSGA-II.
    Otimiza simultaneamente MAE (minimizar) e R2 (maximizar).
    """
    def __init__(self, X_train, y_train, cv=10):
        super().__init__(
            n_var=3,  # variaveis (C, epsilon, gamma)
            n_obj=2,  # fitness (MAE, -R2)
            n_constr=0,
            xl=np.array([1e-3, 1e-4, 1e-4]),  # limites inferiores
            xu=np.array([1e3, 1, 1])  # limites superiores
        )
        self.X_train = X_train
        self.y_train = y_train
        self.cv = cv

    def evaluate_solution(self, x_i):
        """Avalia uma única solução"""
        svr = SVR(
            C=x_i[0],
            epsilon=x_i[1],
            gamma=x_i[2],
            kernel='rbf'
        )
        
        # Calcula MAE e R2 usando validação cruzada
        mae_scores = -cross_val_score(svr, self.X_train, self.y_train, 
                                    scoring='neg_mean_absolute_error', 
                                    cv=self.cv,
                                    n_jobs=1)  # n_jobs=1 pois já estamos paralelizando externamente
        r2_scores = cross_val_score(svr, self.X_train, self.y_train,
                                  scoring='r2',
                                  cv=self.cv,
                                  n_jobs=1)
        
        return np.mean(mae_scores), -np.mean(r2_scores)

    def _evaluate(self, x, out, *args, **kwargs):
        # Paraleliza a avaliação das soluções usando 6 núcleos
        results = Parallel(n_jobs=6)(
            delayed(self.evaluate_solution)(x_i) for x_i in x
        )
        
        # Converte resultados para array
        results = np.array(results)
        mae_values = results[:, 0]
        r2_values = results[:, 1]
        
        out["F"] = np.column_stack([mae_values, r2_values])

    def otimizar(self, pop_size=100, n_gen=10):
        """
        Executa a otimização usando NSGA-II.
        
        Args:
            pop_size: Tamanho da população
            n_gen: Número de gerações
            
        Returns:
            dict: Melhores parâmetros encontrados
            dict: Valores dos objetivos para a melhor solução
        """
        # NSGA-II
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
        
        # Executa a otimização
        res = minimize(
            self,
            algorithm,
            ('n_gen', n_gen),
            verbose=True,
            seed=1
        )
        
        # Obtém a melhor solução de compromisso
        return self._get_best_compromise_solution(res)

    def _get_best_compromise_solution(self, res):
        """
        Seleciona a melhor solução de compromisso da frente de Pareto.
        Usa o método da menor distância ao ponto ideal após normalização.
        """
        # Normaliza os objetivos
        F = res.F
        F_norm = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))
        
        # Calcula a distância ao ponto ideal
        ideal_point = np.zeros(F_norm.shape[1])
        distances = np.linalg.norm(F_norm - ideal_point, axis=1)
        
        # Retorna a solução com menor distância
        best_idx = np.argmin(distances)
        best_params = {
            'C': res.X[best_idx, 0],
            'epsilon': res.X[best_idx, 1],
            'gamma': res.X[best_idx, 2]
        }
        best_objectives = {
            'MAE': res.F[best_idx, 0],
            'R2': -res.F[best_idx, 1]  # Converte de volta para o valor positivo
        }
        
        return best_params, best_objectives

    def criar_modelo(self, params):
        """
        Cria um modelo SVR com os parâmetros otimizados.
        
        Args:
            params: Dicionário com os parâmetros otimizados
            
        Returns:
            SVR: Modelo SVR configurado com os melhores parâmetros
        """
        return SVR(
            C=params['C'],
            epsilon=params['epsilon'],
            gamma=params['gamma'],
            kernel='rbf'
        ) 

def otimizar_svr_grid(X_train, y_train, cv=10, n_jobs=6):
    """
    Otimiza os hiperparâmetros do SVR usando Grid Search.
    
    Args:
        X_train: Features de treino
        y_train: Target de treino
        cv: Número de folds para validação cruzada
        n_jobs: Número de núcleos para paralelização
    """
    # Definir grade de parâmetros
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        'epsilon': [0.01, 0.1, 0.2, 0.5, 1.0],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf']
    }
    
    # Criar validação cruzada aninhada
    inner_cv = KFold(n_splits=10, shuffle=True, random_state=42)
    outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # Converter para numpy array se necessário
    if hasattr(X_train, 'values'):

        X_train_array = X_train.values
    else:
        X_train_array = X_train
        
    if hasattr(y_train, 'values'):
        y_train_array = y_train.values.ravel()
    else:
        y_train_array = y_train
    
    # Criar e configurar o Grid Search com validação cruzada aninhada e paralelização
    grid_search = GridSearchCV(
        estimator=SVR(),
        param_grid=param_grid,
        cv=inner_cv,
        scoring=['r2', 'neg_mean_absolute_error'],
        refit='r2',
        n_jobs=n_jobs,  # Usar 6 núcleos
        verbose=1
    )
    
    # Executar a busca
    print("\nBuscando melhores hiperparâmetros...")
    
    # Realizar validação cruzada externa em paralelo
    def evaluate_fold(train_idx, val_idx):
        X_train_outer, X_val_outer = X_train_array[train_idx], X_train_array[val_idx]
        y_train_outer, y_val_outer = y_train_array[train_idx], y_train_array[val_idx]
        
        # Ajustar o grid search no conjunto de treino externo
        grid_search.fit(X_train_outer, y_train_outer)
        
        # Avaliar no conjunto de validação externo
        y_pred = grid_search.predict(X_val_outer)
        return r2_score(y_val_outer, y_pred), mean_absolute_error(y_val_outer, y_pred)
    
    # Paralelizar a validação cruzada externa
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_fold)(train_idx, val_idx) 
        for train_idx, val_idx in outer_cv.split(X_train_array)
    )
    
    # Extrair scores
    results = np.array(results)
    outer_scores_r2 = results[:, 0]
    outer_scores_mae = results[:, 1]
    
    # Treinar o modelo final com os melhores parâmetros em todos os dados
    final_model = grid_search.fit(X_train_array, y_train_array)
    
    # Imprimir resultados
    print("\nMelhores hiperparâmetros encontrados:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    print(f"\nMelhor score interno (R²): {grid_search.best_score_:.4f}")
    print(f"\nScore médio externo (R²): {np.mean(outer_scores_r2):.4f} (+/- {np.std(outer_scores_r2) * 2:.4f})")
    print(f"Score médio externo (MAE): {np.mean(outer_scores_mae):.4f} (+/- {np.std(outer_scores_mae) * 2:.4f})")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def comparar_otimizacoes_svr(X_train, y_train, X_test, y_test, params_nsga=None):
    """
    Compara os resultados da otimização por Grid Search e NSGA-II.
    
    Args:
        X_train, y_train: Dados de treino
        X_test, y_test: Dados de teste
        params_nsga: Parâmetros já otimizados pelo NSGA-II (opcional)
    """
    # Grid Search
    print("Executando Grid Search...")
    svr_grid, params_grid, _ = otimizar_svr_grid(X_train, y_train)
    y_pred_grid = svr_grid.predict(X_test)
    mae_grid = mean_absolute_error(y_test, y_pred_grid)
    r2_grid = r2_score(y_test, y_pred_grid)
    
    # NSGA-II
    if params_nsga is None:
        print("\nExecutando NSGA-II...")
        optimizer = SVROptimization(X_train, y_train)
        params_nsga, _ = optimizer.otimizar()
    else:
        print("\nUsando parâmetros NSGA-II fornecidos...")
        
    svr_nsga = SVR(
        C=params_nsga['C'],
        epsilon=params_nsga['epsilon'],
        gamma=params_nsga['gamma'],
        kernel='rbf'
    )
    svr_nsga.fit(X_train, y_train)
    y_pred_nsga = svr_nsga.predict(X_test)
    mae_nsga = mean_absolute_error(y_test, y_pred_nsga)
    r2_nsga = r2_score(y_test, y_pred_nsga)
    
    # Plotar comparação de parâmetros
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Comparação de Parâmetros
    plt.subplot(1, 2, 1)
    params_comparison = {
        'Grid Search': params_grid,
        'NSGA-II': params_nsga
    }
    
    param_names = ['C', 'epsilon', 'gamma']
    x = np.arange(len(param_names))
    width = 0.35
    
    plt.bar(x - width/2, [params_grid[p] for p in param_names], width, label='Grid Search')
    plt.bar(x + width/2, [params_nsga[p] for p in param_names], width, label='NSGA-II')
    
    plt.xlabel('Parâmetros')
    plt.ylabel('Valor')
    plt.title('Comparação dos Hiperparâmetros')
    plt.xticks(x, param_names)
    plt.legend()
    plt.yscale('log')
    
    # Plot 2: Comparação de Métricas
    plt.subplot(1, 2, 2)
    metrics = {
        'MAE': [mae_grid, mae_nsga],
        'R²': [r2_grid, r2_nsga]
    }
    
    x = np.arange(len(metrics))
    plt.bar(x - width/2, [metrics[m][0] for m in metrics], width, label='Grid Search')
    plt.bar(x + width/2, [metrics[m][1] for m in metrics], width, label='NSGA-II')
    
    plt.xlabel('Métricas')
    plt.ylabel('Valor')
    plt.title('Comparação das Métricas')
    plt.xticks(x, metrics.keys())
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir resultados
    print("\nResultados Grid Search:")
    print(f"Parâmetros: {params_grid}")
    print(f"MAE: {mae_grid:.4f}")
    print(f"R²: {r2_grid:.4f}")
    
    print("\nResultados NSGA-II:")
    print(f"Parâmetros: {params_nsga}")
    print(f"MAE: {mae_nsga:.4f}")
    print(f"R²: {r2_nsga:.4f}")
    
    return (svr_grid, svr_nsga), (params_grid, params_nsga), (
        {'mae': mae_grid, 'r2': r2_grid},
        {'mae': mae_nsga, 'r2': r2_nsga}
    )

def comparar_hyperparametros_svr(X_train, y_train, params_predefinidos, cv=10):
    """
    Compara os resultados do Grid Search com um conjunto de parâmetros predefinidos usando validação cruzada.
    
    Args:
        X_train, y_train: Dados de treino
        params_predefinidos: Dicionário com parâmetros predefinidos (C, epsilon, gamma)
        cv: Número de folds para validação cruzada
    """
    # Grid Search
    print("Executando Grid Search...")
    svr_grid, params_grid, _ = otimizar_svr_grid(X_train, y_train, cv)
    
    # Modelo com parâmetros predefinidos
    print("\nAvaliando modelo com parâmetros predefinidos...")
    svr_pred = SVR(
        C=params_predefinidos['C'],
        epsilon=params_predefinidos['epsilon'],
        gamma=params_predefinidos['gamma'],
        kernel='rbf'
    )
    
    # Calcular métricas usando validação cruzada
    mae_scores_pred = -cross_val_score(svr_pred, X_train, y_train, 
                                     scoring='neg_mean_absolute_error', 
                                     cv=cv)
    r2_scores_pred = cross_val_score(svr_pred, X_train, y_train,
                                   scoring='r2',
                                   cv=cv)
    
    mae_pred = np.mean(mae_scores_pred)
    r2_pred = np.mean(r2_scores_pred)
    
    # Calcular métricas do Grid Search
    mae_scores_grid = -cross_val_score(svr_grid, X_train, y_train, 
                                     scoring='neg_mean_absolute_error', 
                                     cv=cv)
    r2_scores_grid = cross_val_score(svr_grid, X_train, y_train,
                                   scoring='r2',
                                   cv=cv)
    
    mae_grid = np.mean(mae_scores_grid)
    r2_grid = np.mean(r2_scores_grid)
    
    # Criar figura com 3 subplots
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Comparação de Parâmetros
    plt.subplot(1, 3, 1)
    param_names = ['C', 'epsilon', 'gamma']
    x = np.arange(len(param_names))
    width = 0.35
    
    def convert_gamma(gamma_value): #ajuste para plot (gambiarra)
        if gamma_value == 'scale':
            return 1.0  
        elif gamma_value == 'auto':
            return 0.1  
        return float(gamma_value)
    
    grid_values = [float(params_grid[p]) if p != 'gamma' else convert_gamma(params_grid[p]) for p in param_names]
    pred_values = [float(params_predefinidos[p]) if p != 'gamma' else convert_gamma(params_predefinidos[p]) for p in param_names]
    
    plt.bar(x - width/2, grid_values, width, label='Grid Search')
    plt.bar(x + width/2, pred_values, width, label='Predefinidos')
    
    plt.xlabel('Parâmetros')
    plt.ylabel('Valor')
    plt.title('Comparação dos Hiperparâmetros')
    plt.xticks(x, param_names)
    plt.legend()
    plt.yscale('log')
    
    # Plot 2: Comparação de MAE
    plt.subplot(1, 3, 2)
    x = np.arange(1)
    plt.bar(x - width/2, [mae_grid], width, label='Grid Search')
    plt.bar(x + width/2, [mae_pred], width, label='Predefinidos')
    
    plt.xlabel('Modelos')
    plt.ylabel('MAE')
    plt.title('Comparação do MAE')
    plt.xticks([])
    plt.legend()
    
    # Adicionar valores sobre as barras
    plt.text(x[0] - width/2, mae_grid, f'{mae_grid:.4f}', ha='center', va='bottom')
    plt.text(x[0] + width/2, mae_pred, f'{mae_pred:.4f}', ha='center', va='bottom')
    
    # Plot 3: Comparação de R²
    plt.subplot(1, 3, 3)
    plt.bar(x - width/2, [r2_grid], width, label='Grid Search')
    plt.bar(x + width/2, [r2_pred], width, label='Predefinidos')
    
    plt.xlabel('Modelos')
    plt.ylabel('R²')
    plt.title('Comparação do R²')
    plt.xticks([])
    plt.legend()
    
    # Adicionar valores sobre as barras
    plt.text(x[0] - width/2, r2_grid, f'{r2_grid:.4f}', ha='center', va='bottom')
    plt.text(x[0] + width/2, r2_pred, f'{r2_pred:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # print
    print("\nResultados Grid Search (média da validação cruzada):")
    print(f"Parâmetros: {params_grid}")
    print(f"MAE: {mae_grid:.4f} (+/- {mae_scores_grid.std() * 2:.4f})")
    print(f"R²: {r2_grid:.4f} (+/- {r2_scores_grid.std() * 2:.4f})")
    
    print("\nResultados Parâmetros Predefinidos (média da validação cruzada):")
    print(f"Parâmetros: {params_predefinidos}")
    print(f"MAE: {mae_pred:.4f} (+/- {mae_scores_pred.std() * 2:.4f})")
    print(f"R²: {r2_pred:.4f} (+/- {r2_scores_pred.std() * 2:.4f})")
    
    # Retornar o melhor modelo e seus parâmetros baseado no R²
    if r2_grid > r2_pred:
        return svr_grid, params_grid, {
            'mae': mae_grid,
            'r2': r2_grid,
            'mae_std': mae_scores_grid.std(),
            'r2_std': r2_scores_grid.std()
        }
    else:
        svr_pred.fit(X_train, y_train)  # Treinar o modelo com todos os dados
        return svr_pred, params_predefinidos, {
            'mae': mae_pred,
            'r2': r2_pred,
            'mae_std': mae_scores_pred.std(),
            'r2_std': r2_scores_pred.std()
        }

