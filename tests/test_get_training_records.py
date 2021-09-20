import json

url = '/user/train/history'


def test_get_training_records(client):
    res = client.get(url)
    response_data = json.loads(res.get_data(as_text=True))
    assert res.status_code == 200
    assert response_data['status'] is True
