from fastapi.testclient import TestClient

from src.app.api import app

client = TestClient(app)


# Test pour vérifier la racine de l'API
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "API for articles' summary generation" in response.text


# Test pour vérifier la fonctionnalité de résumé de l'article
def test_summarize_article():
    test_article = "This is a test article to verify the summarization feature."
    response = client.get(f"/summarize/?article={test_article}")
    assert response.status_code == 200
    assert "summary" in response.text
