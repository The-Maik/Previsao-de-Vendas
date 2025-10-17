import streamlit as st
import pandas as pd
import joblib
import os

# --- Configuração da Página ---
st.set_page_config(
    page_title="Previsor de Vendas IA",
    page_icon="📈",
    layout="wide"
)

# --- Funções Auxiliares ---

@st.cache_resource
def load_model():
    """Carrega o modelo treinado do disco."""
    model_path = os.path.join('models', 'sales_predictor_model.joblib')
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        return None

def preprocess_data(df):
    """Prepara os dados para predição, da mesma forma que no treinamento."""
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
    df.dropna(subset=['Data'], inplace=True) # Remove linhas onde a data não pôde ser convertida
    
    df_processed = df.copy()
    df_processed['Ano'] = df_processed['Data'].dt.year
    df_processed['Mes'] = df_processed['Data'].dt.month
    df_processed['Dia'] = df_processed['Data'].dt.day
    df_processed['Dia_da_semana'] = df_processed['Data'].dt.dayofweek
    
    return df_processed, df # Retorna ambos para exibição

# --- Carregar o Modelo ---
model = load_model()

# --- Interface do Usuário ---
st.title('📈 Previsor de Vendas com IA')
st.write(
    "Faça o upload de um arquivo CSV com as datas e informações de promoção "
    "para receber uma previsão de vendas gerada por um modelo de Machine Learning."
)

if model is None:
    st.error(
        "O arquivo do modelo (sales_predictor_model.joblib) não foi encontrado na pasta 'models'. "
        "Por favor, execute o script 'src/train.py' primeiro."
    )
else:
    st.header('1. Faça o Upload dos Dados para Previsão')
    uploaded_file = st.file_uploader(
        "Seu arquivo CSV deve conter as colunas 'Data' e 'Promocao_Ativa'.",
        type=['csv']
    )

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            
            st.header("2. Dados Carregados")
            st.dataframe(input_df)

            if st.button("🚀 Realizar Previsão", type="primary"):
                with st.spinner('Processando dados e fazendo previsões...'):
                    
                    # Pré-processar os dados
                    processed_df, original_df_with_date = preprocess_data(input_df)

                    # Selecionar as mesmas features do treinamento
                    features = ['Ano', 'Mes', 'Dia', 'Dia_da_semana', 'Promocao_Ativa']
                    X_to_predict = processed_df[features]

                    # Fazer a previsão
                    predictions = model.predict(X_to_predict)

                    # Adicionar as previsões ao DataFrame original para exibição
                    result_df = original_df_with_date.copy()
                    result_df['Vendas_Previstas'] = predictions
                    result_df['Vendas_Previstas'] = result_df['Vendas_Previstas'].round(2)


                st.header("✅ Previsões Concluídas!")
                st.dataframe(result_df)

                # Criar um gráfico com as previsões
                st.subheader("Gráfico das Vendas Previstas")
                
                # Certificar que a data está ordenada para o gráfico de linha
                result_df_sorted = result_df.sort_values(by='Data')
                
                st.line_chart(result_df_sorted, x='Data', y='Vendas_Previstas')

        except Exception as e:
            st.error(f"Ocorreu um erro ao processar o arquivo: {e}")