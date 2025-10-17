import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os


def train_model():
    """
    Função para carregar os dados, treinar um modelo de regressão 
    e salvá-lo em disco.
    """
    print("Iniciando o processo de treinamento...")

    # 1 Carregar os dados
    try:
        data_path = os.path.join(os.path.dirname(
            __file__), '..', 'data', 'raw', 'historical_sales.csv')
        df = pd.read_csv(data_path, parse_dates=['Data'])
        print("Dados carregados com sucesso.")
    except FileNotFoundError:
        print(
            f"Erro: Arquivo não encontrado em {data_path}. Certifique-se de que 'generate_data.py' foi executado.")
        return

    # 2. Preparar os dados (Engenharia de Features)
    # Modelos de ML não entendem datas, então transformamos a data em números
    df['Ano'] = df['Data'].dt.year
    df['Mes'] = df['Data'].dt.month
    df['Dia'] = df['Data'].dt.day
    df['Dia_da_semana'] = df['Data'].dt.dayofweek  # Segunda=0, Domingo=6

    # Remover a coluna de data original, pois já extraímos as informações
    df = df.drop('Data', axis=1)
    print("Engenharia de features concluída.")

    # 3. Definir features (X) e o alvo (y)
    features = ['Ano', 'Mes', 'Dia', 'Dia_da_semana', 'Promocao_Ativa']
    target = 'Vendas'

    X = df[features]
    y = df[target]

    # 4. Dividir os dados em treino e teste
    # Usamos random_state=42 para garantir que a divisão seja sempre a mesma (reprodutibilidade)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print("Dados divididos em conjuntos de treino e teste.")

    # 5. Treinar o modelo
    # RandomForestRegressor é um bom modelo para começar: poderoso e versátil
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    print("Treinando o modelo RandomForestRegressor...")
    model.fit(X_train, y_train)
    print("Modelo treinado com sucesso!")

    # 6. Salvar o modelo treinado
    # Criar a pasta 'models' se ela não existir
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, 'sales_predictor_model.joblib')
    joblib.dump(model, model_path)
    print(f"Modelo salvo com sucesso em: {model_path}")


if __name__ == '__main__':
    train_model()
