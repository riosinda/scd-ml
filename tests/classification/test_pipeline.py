from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scd_ml.classification.columns import CLASS_ORDER, encode_target
from scd_ml.classification.pipeline import (
    MODELS,
    MetadataEncoder,
    ModelConfig,
    RBFFeatureMap,
    build_selector,
    fit_full_pipeline,
    fit_params,
    model_input_columns,
)

from .synthetic import synthetic_cohort


class MetadataEncoderTests(unittest.TestCase):
    def test_missing_and_unseen_values_are_encoded_from_train_only(self) -> None:
        training = pd.DataFrame(
            {
                "age_approx": [30.0, 50.0, np.nan],
                "sex": ["male", None, "female"],
                "anatom_site_1": ["Trunk", "Head and neck", None],
            }
        )
        validation = pd.DataFrame(
            {"age_approx": [np.nan], "sex": ["other"], "anatom_site_1": ["Trunk"]}
        )

        encoder = MetadataEncoder().fit(training)
        encoded = encoder.transform(validation).iloc[0]

        self.assertEqual(encoded["age_approx"], 40.0)
        self.assertEqual(encoded["age_approx__missing"], 1.0)
        self.assertEqual(encoded[["sex__female", "sex__male", "sex__missing"]].sum(), 0.0)
        self.assertEqual(encoded["anatom_site_1__trunk"], 1.0)
        self.assertIn("anatom_site_1__head_and_neck", encoder.get_feature_names_out())


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        cohort = synthetic_cohort()
        eligible = cohort[cohort["eligible_for_classification"]]
        self.development = eligible[eligible["split"].eq("train")].reset_index(drop=True)
        self.test = eligible[eligible["split"].eq("test")].reset_index(drop=True)
        self.y = encode_target(self.development["target"])

    def fit(self, **overrides: object):
        values = {
            "channel_set": "all",
            "use_metadata": False,
            "selection": "none",
            "balancing": "none",
            "model": "logreg",
        }
        values.update(overrides)
        config = ModelConfig(**values)
        columns = model_input_columns(self.development.columns, config)
        pipeline = fit_full_pipeline(config, self.development[columns], self.y)
        return pipeline, pipeline.predict_proba(self.test[columns])

    def test_every_model_predicts_four_class_probabilities(self) -> None:
        for model in MODELS:
            with self.subTest(model=model):
                params = {"n_estimators": 20} if model in {"xgboost", "random_forest"} else {}
                pipeline, proba = self.fit(model=model, params=params)
                self.assertEqual(proba.shape, (len(self.test), len(CLASS_ORDER)))
                np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)
                self.assertEqual(pipeline.steps[-1][1].classes_.tolist(), [0, 1, 2, 3])

    def test_selectors_and_balancers_combine_with_metadata(self) -> None:
        for selection in ("anova", "l1", "rfe"):
            for balancing in ("class_weight", "smote"):
                with self.subTest(selection=selection, balancing=balancing):
                    pipeline, proba = self.fit(
                        selection=selection,
                        balancing=balancing,
                        use_metadata=True,
                        k=5,
                        model="mlp" if balancing == "class_weight" else "logreg",
                    )
                    selector = pipeline.named_steps["selector"]
                    self.assertEqual(int(selector.get_support().sum()), 5)
                    self.assertEqual(proba.shape[1], len(CLASS_ORDER))

    def test_class_weight_becomes_sample_weight_for_models_without_it(self) -> None:
        y = np.array([0, 0, 0, 1, 2, 3])
        for model in ("xgboost", "mlp"):
            config = ModelConfig("gray", False, "none", "class_weight", model)
            weights = fit_params(config, y)["model__sample_weight"]
            self.assertAlmostEqual(weights[0] * 3, weights[3])
        config = ModelConfig("gray", False, "none", "class_weight", "logreg")
        self.assertEqual(fit_params(config, y), {})

    def test_rbf_gamma_scales_with_the_number_of_selected_features(self) -> None:
        X = np.random.default_rng(0).normal(size=(30, 8))
        feature_map = RBFFeatureMap(gamma_scale=2.0, n_components=100).fit(X)
        self.assertAlmostEqual(feature_map.nystroem_.gamma, 0.25)
        self.assertEqual(feature_map.transform(X).shape, (30, 30))

    def test_selector_k_is_capped_at_available_features(self) -> None:
        selector = build_selector("anova", 500, n_features=12, seed=0)
        self.assertEqual(selector.k, 12)

    def test_smote_only_resamples_during_fit(self) -> None:
        pipeline, proba = self.fit(balancing="smote")
        self.assertEqual(len(proba), len(self.test))

    def test_config_round_trips_through_dict(self) -> None:
        config = ModelConfig("gray", True, "anova", "smote", "xgboost", k=10, params={"a": 1})
        self.assertEqual(ModelConfig.from_dict(config.to_dict()), config)
        with self.assertRaises(ValueError):
            ModelConfig("gray", False, "anova", "none", "logreg", k=None)


if __name__ == "__main__":
    unittest.main()
