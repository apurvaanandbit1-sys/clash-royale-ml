import unittest
import numpy as np
import tempfile
from pathlib import Path
from training.baselines import (
    RandomGuessModel, MajorityClassModel, LogisticRegressionModel,
    RandomForestModel, LightGBMModel, CatBoostModel, MLPModel
)

class TestCRBaselineModels(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create a synthetic dataset of 100 samples and 10 features
        self.X_train = np.random.choice([0.0, 1.0], size=(100, 10))
        self.y_train = np.random.choice([0.0, 1.0], size=100)
        
        self.X_test = np.random.choice([0.0, 1.0], size=(20, 10))
        self.y_test = np.random.choice([0.0, 1.0], size=20)
        
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_model_shapes_and_evaluation(self):
        models = [
            RandomGuessModel(),
            MajorityClassModel(),
            LogisticRegressionModel(max_iter=10),
            RandomForestModel(n_estimators=5, max_depth=3),
            LightGBMModel(n_estimators=5, max_depth=3),
            CatBoostModel(iterations=5, depth=3),
            MLPModel(max_iter=10)
        ]

        for model in models:
            model.fit(self.X_train, self.y_train)
            
            preds = model.predict(self.X_test)
            probs = model.predict_proba(self.X_test)
            
            # Check shapes
            self.assertEqual(preds.shape, (20,))
            self.assertEqual(probs.shape, (20,))
            
            # Check values in bounds
            self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))
            
            # Evaluate metrics dict
            metrics = model.evaluate(self.X_test, self.y_test)
            self.assertIn("accuracy", metrics)
            self.assertIn("roc_auc", metrics)
            self.assertIn("log_loss", metrics)
            self.assertIn("ece", metrics)

    def test_serialization(self):
        model = LogisticRegressionModel()
        model.fit(self.X_train, self.y_train)
        
        orig_preds = model.predict(self.X_test)
        
        # Save model
        filepath = Path(self.temp_dir.name) / "test_model.pkl"
        model.save(str(filepath))
        
        # Load model into new instance
        loaded_model = LogisticRegressionModel()
        loaded_model.load(str(filepath))
        
        loaded_preds = loaded_model.predict(self.X_test)
        
        # Assert predictions match exactly
        np.testing.assert_array_equal(orig_preds, loaded_preds)

    def test_reproducibility(self):
        # Two identical RandomForests fitted with seed 42 should give identical outputs
        rf1 = RandomForestModel(n_estimators=10, random_seed=42)
        rf1.fit(self.X_train, self.y_train)
        
        rf2 = RandomForestModel(n_estimators=10, random_seed=42)
        rf2.fit(self.X_train, self.y_train)
        
        np.testing.assert_allclose(rf1.predict_proba(self.X_test), rf2.predict_proba(self.X_test), rtol=1e-6, atol=1e-6)

if __name__ == "__main__":
    unittest.main()
