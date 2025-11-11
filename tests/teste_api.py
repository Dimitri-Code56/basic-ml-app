import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# Importa o app FastAPI do seu módulo
from app.app import app, collection 
# Nota: A importação acima pode falhar se o app.py tenta carregar IntentClassifier antes
# que os mocks sejam aplicados. Caso falhe, mude o escopo dos mocks.

# 1. FIXTURE BÁSICA PARA O CLIENTE DE TESTE
@pytest.fixture(scope="module")
def client():
    """Retorna um TestClient do FastAPI para fazer requisições."""
    # O TestClient é leve e rápido, ideal para testes de API.
    return TestClient(app)

# 2. MOCKS PARA O MODELO ML E O BANCO DE DADOS
# Mockar o IntentClassifier para que o carregamento do modelo não seja real
@pytest.fixture(autouse=True)
def mock_ml_logic():
    """Simula o IntentClassifier e seu método predict."""
    
    # Simula a predição retornando uma intenção e probabilidades fixas
    mock_model_instance = MagicMock()
    mock_model_instance.predict.return_value = ("saudacao", {"saudacao": 0.95, "outro": 0.05})
    
    # Simula a classe IntentClassifier
    mock_classifier_class = MagicMock(return_value=mock_model_instance)
    
    with patch('app.app.IntentClassifier', mock_classifier_class), \
         patch('app.app.os.listdir', return_value=['dummy_model.keras']): # Simula o arquivo de modelo
        yield # O teste será executado com os mocks ativos

# Mockar o DB para evitar escrita real no MongoDB
@pytest.fixture(autouse=True)
def mock_db_connection():
    """Simula a coleção do MongoDB para testes de escrita."""
    mock_collection = MagicMock()
    with patch('app.app.get_mongo_collection', return_value=mock_collection):
        yield mock_collection

# --- TESTES FUNCIONAIS (API ENDPOINTS) ---

## Testes no Endpoint Raiz (Disponibilidade)
def test_root_endpoint(client):
    """Verifica se o endpoint raiz está funcionando."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Basic ML App is running" in response.json()["message"]

## Testes no Endpoint de Predição (/predict)

def test_predict_success_in_dev_mode(client, mock_db_connection, monkeypatch):
    """Testa a predição e o logging no modo 'dev' (sem auth)."""
    # 1. Configura o ambiente para 'dev'
    monkeypatch.setenv("ENV", "dev")
    
    # 2. Faz a requisição
    response = client.post("/predict?text=olá")
    
    # 3. Asserts da Resposta
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "olá"
    assert data["owner"] == "dev_user" # Autenticação ignorada
    assert data["predictions"]["dummy_model"]["top_intent"] == "saudacao" # Resultado do Mock
    
    # 4. Asserts do Banco de Dados (Integração)
    # Verifica se a função insert_one foi chamada UMA vez
    mock_db_connection.insert_one.assert_called_once()
    
def test_predict_requires_auth_in_prod_mode(client, monkeypatch):
    """Testa se a autenticação é imposta no modo 'prod'."""
    # 1. Configura o ambiente para 'prod'
    monkeypatch.setenv("ENV", "prod")
    
    # 2. Mocka a função de autenticação (verify_token) para falhar,
    # simulando a ausência do token
    with patch('app.app.verify_token', side_effect=Exception("Token missing")):
        response = client.post("/predict?text=olá")
        
    # 3. Assert da Resposta
    assert response.status_code == 401
    assert "Authentication failed" in response.json()["detail"]

def test_predict_with_valid_auth_in_prod_mode(client, monkeypatch, mock_db_connection):
    """Testa se a predição é bem-sucedida com token válido no modo 'prod'."""
    # 1. Configura o ambiente para 'prod'
    monkeypatch.setenv("ENV", "prod")
    
    # 2. Mocka a função de autenticação para retornar um usuário válido
    with patch('app.app.verify_token', return_value="prod_user_123"):
        response = client.post("/predict?text=olá")
        
    # 3. Assert da Resposta
    assert response.status_code == 200
    data = response.json()
    assert data["owner"] == "prod_user_123"
    
    # 4. Asserts do Banco de Dados
    mock_db_connection.insert_one.assert_called_once()
    
def test_predict_handles_db_logging_failure(client, monkeypatch, mock_db_connection):
    """Testa se a falha no DB não impede a resposta da API."""
    # 1. Configura o ambiente para 'dev'
    monkeypatch.setenv("ENV", "dev")

    # 2. Mocka a inserção do DB para levantar uma exceção
    mock_db_connection.insert_one.side_effect = Exception("DB connection error")

    # 3. Faz a requisição
    response = client.post("/predict?text=olá")

    # 4. Assert da Resposta
    assert response.status_code == 200 # A API deve responder mesmo com erro no log
    data = response.json()
    assert data.get("db_error") == "Failed to log prediction to database."