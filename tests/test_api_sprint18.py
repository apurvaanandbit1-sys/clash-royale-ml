import sys
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from api.app import app

class TestMatchupAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["onnx_model_loaded"])

    def test_predict_endpoint(self):
        deck_a = [26000000, 26000001, 26000002, 26000003, 26000004, 26000005, 26000006, 26000007]
        deck_b = [26000008, 26000009, 26000010, 26000011, 26000012, 26000013, 26000014, 26000015]
        
        payload = {
            "p1_deck": deck_a,
            "p2_deck": deck_b,
            "trophy_diff": 0.0
        }
        
        # First call (uncached)
        resp1 = self.client.post("/predict", json=payload)
        self.assertEqual(resp1.status_code, 200)
        data1 = resp1.json()
        self.assertIn("win_probability", data1)
        self.assertIn("categorical_band", data1)
        self.assertIn(data1["categorical_band"], [
            "Strongly Favored", "Slightly Favored", "Even Matchup", "Slightly Unfavored", "Strongly Unfavored"
        ])
        self.assertFalse(data1["cached"])
        
        # Second call (cached hit)
        resp2 = self.client.post("/predict", json=payload)
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.json()
        self.assertTrue(data2["cached"])
        self.assertEqual(data1["win_probability"], data2["win_probability"])

    def test_embeddings_endpoint(self):
        response = self.client.get("/embeddings")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["checkpoint"], "sprint13_matchup_model.pt")
        self.assertEqual(data["projection_method"], "t-SNE")
        self.assertGreater(data["num_cards"], 0)
        self.assertIn("coordinates", data)

if __name__ == "__main__":
    unittest.main()
