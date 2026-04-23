from app import app


def test_health_endpoint():
    client = app.test_client()
    response = client.get("/health")

    assert response.status_code == 200

    data = response.get_json()
    assert data["status"] == "ok"
    assert "classes" in data
    assert isinstance(data["classes"], list)


def test_detect_requires_login():
    client = app.test_client()
    response = client.get("/detect", follow_redirects=False)

    assert response.status_code in (301, 302)
    assert "/?popup=login" in response.headers["Location"]


def test_predict_without_image_returns_400():
    client = app.test_client()
    response = client.post("/predict", data={}, follow_redirects=False)

    assert response.status_code == 400

    data = response.get_json()
    assert "error" in data