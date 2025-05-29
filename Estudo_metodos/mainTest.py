from modelos import *
import matplotlib.pyplot as plt
import seaborn as sns

def plot_svr_summary():
    # Configuração visual
    plt.style.use('default')  # Mudando de seaborn para default
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # dados para o grafico
    approaches = ['NSGA-II', 'Grid Search', 'Predefinido']
    param_ranges = {
        'C': ['1e-3 - 1e3', '1e-3 - 1e3', '999.99'],
        'epsilon': ['1e-4 - 1', '0.01 - 1.0', '0.00056'],
        'gamma': ['1e-4 - 1', '1e-3 - 0.1', '0.073']
    }
    
    # Plot range hiperparâmetros
    x = np.arange(len(param_ranges))
    width = 0.25
    
    for i, approach in enumerate(approaches):
        values = [param_ranges[param][i] for param in param_ranges.keys()]
        ax1.bar(x + i*width - width, [1]*3, width, label=approach, alpha=0.7)
        
        # Adicionar texto nas barras
        for j, value in enumerate(values):
            ax1.text(j + i*width - width, 0.5, value,
                    ha='center', va='center', rotation=90, fontsize=8)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(param_ranges.keys())
    ax1.set_title('Ranges dos Hiperparâmetros por Abordagem')
    ax1.set_ylim(0, 1.2)
    ax1.set_yticks([])
    ax1.legend()
    
    features = {
        'NSGA-II': ['Multi-objetivo', 'População: 100', 'Gerações: 10', 'CV: 10-fold'],
        'Grid Search': ['Busca exaustiva', 'CV aninhado', 'Inner: 10-fold', 'Outer: 10-fold'],
        'Predefinido': ['Valores fixos', 'Baseado em NSGA-II', 'Otimização prévia', 'CV: 10-fold']
    }
    
    y_pos = np.arange(len(features))
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(features.keys())
    
    for i, (approach, chars) in enumerate(features.items()):
        ax2.text(0.1, i, ' | '.join(chars), fontsize=9, va='center')
    
    ax2.set_title('Características das Abordagens')
    ax2.set_xlim(0, 1)
    ax2.set_xticks([])
    
    plt.tight_layout()
    plt.show()

def code():
    try:
        X_train, X_test, y_train, y_test, feature_names = carregar_e_preparar_dados()
        
        modelos = [
            RegressaoLinear(),
            SVR_Model(),
            RandomForest()
        ]
        
        # avalia cada modelos
        for modelo in modelos:
            modelo.treinar(X_train, y_train)
            modelo.validacao_cruzada(X_train, y_train)
            modelo.avaliar(X_test, y_test)
        
        # plota features relevantes
        for modelo in modelos:
            if isinstance(modelo, RandomForest):
                modelo.plotar_importancia_features(feature_names)
        
        # Adiciona o plot resumo do SVR
        plot_svr_summary()

        # Compara os modelos usando a função da classe modelos
        from modelos import comparar_modelos
        comparacao = comparar_modelos(modelos)
        
        return modelos, comparacao
    
    except Exception as e:
        print(f"\nErro durante o treinamento: {str(e)}")
        import traceback
        print("\nDetalhes do erro:")
        print(traceback.format_exc())
        return None, None

if __name__ == "__main__":
    try:
        print("=== Inicio ===")
        modelos, comparacao = code()
        print("\nProcesso finalizado com sucesso!")
    except Exception as e:
        print(f"\nErro durante a execução: {str(e)}")
        import traceback
        print("\nDetalhes do erro:")
        print(traceback.format_exc()) 
