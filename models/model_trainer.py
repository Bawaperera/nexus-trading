"""
NEXUS — Phase 2: ML Model Trainer
Trains an XGBoost ensemble using walk-forward validation to avoid lookahead bias.

Walk-forward validation:
  Fold 1: Train on rows 0–325  → Test on rows 326–651
  Fold 2: Train on rows 0–651  → Test on rows 652–977
  Fold 3: Train on rows 0–977  → Test on rows 978–1302
  ...
  Each fold always trains on PAST data and tests on FUTURE data.
  Never the other way around (that's data snooping / lookahead bias).

Class encoding:
  XGBoost needs 0-indexed classes.
  SELL(-1) → 0  |  HOLD(0) → 1  |  BUY(1) → 2
"""

import os, logging, warnings
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (classification_report, accuracy_score,
                              confusion_matrix, f1_score)
warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains, validates, and saves the NEXUS XGBoost model.

    Usage:
        trainer = ModelTrainer()
        results = trainer.walk_forward_validate(X, y)
        model   = trainer.train_final(X, y)
        trainer.save(model, "models/nexus_xgb.pkl")
    """

    # Hyperparameters tuned for financial time-series
    # Intentionally conservative to prevent overfitting
    XGB_PARAMS = {
        "n_estimators":      300,
        "max_depth":         4,       # Shallow trees = less overfitting
        "learning_rate":     0.03,    # Slow learning = better generalization
        "subsample":         0.80,    # Use 80% of rows per tree
        "colsample_bytree":  0.70,    # Use 70% of features per tree
        "min_child_weight":  5,       # Min samples in leaf node
        "gamma":             0.20,    # Min gain to split a node
        "reg_alpha":         0.10,    # L1 regularization
        "reg_lambda":        1.50,    # L2 regularization
        "objective":         "multi:softprob",
        "num_class":         3,       # SELL, HOLD, BUY
        "eval_metric":       "mlogloss",
        "use_label_encoder": False,
        "random_state":      42,
        "n_jobs":            -1,
        "verbosity":         0,
    }

    # Class mapping: original → XGBoost index
    CLASS_MAP     = {-1: 0, 0: 1, 1: 2}
    CLASS_MAP_INV = {0: -1, 1: 0, 2: 1}
    CLASS_NAMES   = {0: "SELL", 1: "HOLD", 2: "BUY"}

    def __init__(self, n_splits: int = 5, model_dir: str = "models"):
        self.n_splits  = n_splits
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        log.info(f"ModelTrainer initialized | Walk-forward splits: {n_splits}")

    # ─── Public API ───────────────────────────────────────────────────────────

    def walk_forward_validate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Run walk-forward cross-validation and return full out-of-sample predictions.

        Returns:
            dict with keys:
              oof_predictions  → Series of out-of-sample predicted classes (-1,0,1)
              oof_probabilities → DataFrame of out-of-sample probabilities
              fold_metrics     → list of per-fold performance dicts
              overall_metrics  → aggregated performance across all folds
              feature_importance → mean feature importance across all folds
        """
        log.info(f"\n{'='*55}")
        log.info(f"WALK-FORWARD VALIDATION | {len(X)} samples | {self.n_splits} folds")
        log.info(f"{'='*55}")

        y_encoded = y.map(self.CLASS_MAP)
        tscv      = TimeSeriesSplit(n_splits=self.n_splits)

        oof_preds  = pd.Series(index=y.index, dtype=float)
        oof_probas = pd.DataFrame(index=y.index, columns=["SELL", "HOLD", "BUY"], dtype=float)
        fold_metrics   = []
        importances    = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_encoded.iloc[train_idx], y_encoded.iloc[test_idx]

            model = XGBClassifier(**self.XGB_PARAMS)
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

            proba = model.predict_proba(X_test)
            preds = np.argmax(proba, axis=1)

            # Store out-of-fold results
            oof_preds.iloc[test_idx]           = [self.CLASS_MAP_INV[p] for p in preds]
            oof_probas.iloc[test_idx, 0]       = proba[:, 0]  # SELL
            oof_probas.iloc[test_idx, 1]       = proba[:, 1]  # HOLD
            oof_probas.iloc[test_idx, 2]       = proba[:, 2]  # BUY

            # Per-fold metrics
            acc     = accuracy_score(y_test, preds)
            f1      = f1_score(y_test, preds, average="weighted", zero_division=0)
            metrics = {
                "fold":          fold,
                "train_samples": len(train_idx),
                "test_samples":  len(test_idx),
                "accuracy":      round(acc, 4),
                "f1_weighted":   round(f1, 4),
            }
            fold_metrics.append(metrics)
            importances.append(dict(zip(X.columns, model.feature_importances_)))

            log.info(
                f"  Fold {fold}/{self.n_splits} | Train: {len(train_idx):4d} | "
                f"Test: {len(test_idx):4d} | Acc: {acc:.3f} | F1: {f1:.3f}"
            )

        # Drop rows that never got predictions (first fold's train set)
        valid_mask = oof_preds.notna()
        oof_preds  = oof_preds[valid_mask].astype(int)
        oof_probas = oof_probas[valid_mask]
        y_valid    = y[valid_mask]

        overall = self._compute_overall_metrics(y_valid, oof_preds)
        feat_imp = pd.DataFrame(importances).mean().sort_values(ascending=False)

        self._print_report(overall, fold_metrics, feat_imp)

        return {
            "oof_predictions":   oof_preds,
            "oof_probabilities": oof_probas,
            "fold_metrics":      fold_metrics,
            "overall_metrics":   overall,
            "feature_importance": feat_imp,
        }

    def train_final(self, X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
        """
        Train the final model on ALL data (for live trading).
        Use only after walk-forward validation confirms profitability.
        """
        log.info(f"\nTraining final model on {len(X)} samples (all folds)...")
        y_encoded = y.map(self.CLASS_MAP)
        model     = XGBClassifier(**self.XGB_PARAMS)
        model.fit(X, y_encoded, verbose=False)
        log.info("Final model trained ✅")
        return model

    def predict(self, model: XGBClassifier, X: pd.DataFrame) -> dict:
        """
        Run inference on new data. Returns probabilities for each class.

        Returns:
            {
                "action": "BUY" | "SELL" | "HOLD",
                "BUY": 0.72,
                "SELL": 0.18,
                "HOLD": 0.10,
                "confidence": 0.72
            }
        """
        proba   = model.predict_proba(X.iloc[[-1]])[0]
        buy_p   = float(proba[2])   # class index 2 = BUY
        sell_p  = float(proba[0])   # class index 0 = SELL
        hold_p  = float(proba[1])   # class index 1 = HOLD

        best_idx = np.argmax(proba)
        action   = self.CLASS_NAMES[best_idx]

        return {
            "action":     action,
            "BUY":        round(buy_p, 4),
            "SELL":       round(sell_p, 4),
            "HOLD":       round(hold_p, 4),
            "confidence": round(max(buy_p, sell_p, hold_p), 4),
        }

    def save(self, model: XGBClassifier, filename: str = "nexus_xgb.pkl"):
        path = os.path.join(self.model_dir, filename)
        joblib.dump(model, path)
        log.info(f"Model saved → {path}")
        return path

    def load(self, filename: str = "nexus_xgb.pkl") -> XGBClassifier:
        path = os.path.join(self.model_dir, filename)
        model = joblib.load(path)
        log.info(f"Model loaded ← {path}")
        return model

    # ─── Private helpers ──────────────────────────────────────────────────────

    def _compute_overall_metrics(self, y_true, y_pred) -> dict:
        acc   = accuracy_score(y_true, y_pred)
        f1    = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        buy_f1 = f1_score(
            (y_true == 1).astype(int),
            (y_pred == 1).astype(int),
            zero_division=0
        )
        sell_f1 = f1_score(
            (y_true == -1).astype(int),
            (y_pred == -1).astype(int),
            zero_division=0
        )

        dist = pd.Series(y_pred).value_counts()
        return {
            "accuracy":       round(acc, 4),
            "f1_weighted":    round(f1, 4),
            "f1_buy":         round(buy_f1, 4),
            "f1_sell":        round(sell_f1, 4),
            "n_buy_signals":  int(dist.get(1, 0)),
            "n_sell_signals": int(dist.get(-1, 0)),
            "n_hold_signals": int(dist.get(0, 0)),
            "total_signals":  len(y_pred),
        }

    def _print_report(self, overall, fold_metrics, feat_imp):
        log.info(f"\n{'='*55}")
        log.info("WALK-FORWARD RESULTS")
        log.info(f"{'='*55}")
        log.info(f"  Overall Accuracy : {overall['accuracy']:.4f} ({overall['accuracy']*100:.2f}%)")
        log.info(f"  Weighted F1      : {overall['f1_weighted']:.4f}")
        log.info(f"  BUY  F1-score    : {overall['f1_buy']:.4f}")
        log.info(f"  SELL F1-score    : {overall['f1_sell']:.4f}")
        log.info(f"  Signals → BUY: {overall['n_buy_signals']} | SELL: {overall['n_sell_signals']} | HOLD: {overall['n_hold_signals']}")
        log.info(f"\n  Top 10 most important features:")
        for feat, imp in feat_imp.head(10).items():
            bar = "█" * int(imp * 200)
            log.info(f"    {feat:30s} {imp:.4f}  {bar}")
