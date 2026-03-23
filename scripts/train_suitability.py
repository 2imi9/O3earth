#!/usr/bin/env python3
"""
Train a lightweight suitability classifier on OlmoEarth embeddings.

Loads pre-extracted embeddings (768-dim) + labels, trains XGBoost and MLP
classifiers, evaluates with AUC-ROC / Precision@K / Recall@K, and runs
ablation studies comparing embeddings-only vs embeddings + external features.

Usage:
    python scripts/train_suitability.py \
        --embeddings data/embeddings/embeddings.npy \
        --metadata data/embeddings/embeddings_meta.csv \
        --output-dir results/suitability \
        --energy-type solar \
        --test-size 0.2 \
        --seed 42
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, k_values: list[int] = None) -> dict:
    """Compute AUC-ROC, AP, Precision@K, Recall@K."""
    if k_values is None:
        k_values = [50, 100, 200, 500]

    metrics = {}

    # AUC-ROC
    try:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc_roc"] = 0.0

    # Average Precision (area under PR curve)
    try:
        metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["average_precision"] = 0.0

    # Accuracy at threshold=0.5
    y_pred = (y_prob >= 0.5).astype(int)
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    # Precision@K, Recall@K
    sorted_idx = np.argsort(-y_prob)  # descending
    total_pos = int(y_true.sum())

    for k in k_values:
        if k > len(y_true):
            continue
        top_k = sorted_idx[:k]
        tp_at_k = int(y_true[top_k].sum())
        metrics[f"precision_at_{k}"] = tp_at_k / k if k > 0 else 0.0
        metrics[f"recall_at_{k}"] = tp_at_k / total_pos if total_pos > 0 else 0.0

    # Calibration (5 bins)
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5, strategy="uniform")
        metrics["calibration"] = {
            "predicted": prob_pred.tolist(),
            "actual": prob_true.tolist(),
        }
    except Exception:
        metrics["calibration"] = None

    return metrics


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def train_xgboost(X_train, y_train, X_test, y_test, seed=42) -> tuple[dict, object]:
    """Train XGBoost classifier, return metrics and model."""
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("xgboost not installed. Install: pip install xgboost")
        return {}, None

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=seed,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_prob)
    metrics["model"] = "xgboost"

    # Feature importance (top 20)
    importances = model.feature_importances_
    top_idx = np.argsort(-importances)[:20]
    metrics["top_features"] = [
        {"index": int(i), "importance": float(importances[i])} for i in top_idx
    ]

    return metrics, model


def train_mlp(X_train, y_train, X_test, y_test, seed=42, epochs=100) -> tuple[dict, object]:
    """Train a simple MLP classifier using sklearn, return metrics."""
    from sklearn.neural_network import MLPClassifier

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=epochs,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        verbose=False,
    )

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_prob)
    metrics["model"] = "mlp"

    return metrics, model


def train_logistic(X_train, y_train, X_test, y_test, seed=42) -> tuple[dict, object]:
    """Train logistic regression baseline."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        C=1.0, max_iter=500, random_state=seed, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_prob)
    metrics["model"] = "logistic_regression"

    return metrics, model


# ---------------------------------------------------------------------------
# Ablation: synthesize simple external features
# ---------------------------------------------------------------------------


def _synthesize_external_features(meta_df: pd.DataFrame) -> np.ndarray:
    """Create simple external features from metadata for ablation study.

    These are proxy features that would come from external APIs in production:
    - abs(latitude) — proxy for solar irradiance / climate zone
    - cos(latitude_rad) — area weighting
    - longitude (normalized) — rough east/west positioning
    - latitude^2 — non-linear latitude effect

    In a real system, these would be replaced with actual GHI, wind speed,
    grid distance, elevation, slope, etc.
    """
    lat = meta_df["lat"].values
    lon = meta_df["lon"].values

    features = np.column_stack([
        np.abs(lat) / 90.0,                    # normalized abs latitude
        np.cos(np.radians(lat)),                # cos(lat)
        lon / 180.0,                            # normalized longitude
        (lat / 90.0) ** 2,                      # lat^2
        np.sin(np.radians(lon)),                # sin(lon)
        np.cos(np.radians(lon)),                # cos(lon)
        np.abs(lat - 35) / 90.0,               # distance from "solar belt"
        (np.abs(lat) > 30).astype(float),       # high latitude flag
    ])

    return features.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train suitability classifier on OlmoEarth embeddings."
    )
    parser.add_argument(
        "--embeddings",
        default="data/embeddings/embeddings.npy",
        help="Path to embeddings.npy file.",
    )
    parser.add_argument(
        "--metadata",
        default="data/embeddings/embeddings_meta.csv",
        help="Path to embeddings_meta.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/suitability",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--energy-type",
        default=None,
        help="Filter to specific energy type (solar, wind). None = use all.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--cv-folds", type=int, default=0, help="If >0, run K-fold cross-validation.")
    parser.add_argument("--skip-xgboost", action="store_true", help="Skip XGBoost training.")
    parser.add_argument("--skip-mlp", action="store_true", help="Skip MLP training.")
    parser.add_argument("--epochs", type=int, default=100, help="MLP max epochs.")

    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).resolve().parent.parent
    embeddings_path = Path(args.embeddings)
    if not embeddings_path.is_absolute():
        embeddings_path = project_root / embeddings_path
    meta_path = Path(args.metadata)
    if not meta_path.is_absolute():
        meta_path = project_root / meta_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    logger.info(f"Loading embeddings from {embeddings_path} ...")
    embeddings = np.load(str(embeddings_path))
    logger.info(f"  Shape: {embeddings.shape}")

    logger.info(f"Loading metadata from {meta_path} ...")
    meta_df = pd.read_csv(str(meta_path))
    logger.info(f"  {len(meta_df)} rows")

    assert len(embeddings) == len(meta_df), (
        f"Mismatch: {len(embeddings)} embeddings vs {len(meta_df)} metadata rows"
    )

    # --- Filter by energy type ---
    if args.energy_type:
        mask = meta_df["energy_type"].str.lower() == args.energy_type.lower()
        embeddings = embeddings[mask.values]
        meta_df = meta_df[mask].reset_index(drop=True)
        logger.info(f"Filtered to {args.energy_type}: {len(meta_df)} samples")

    labels = meta_df["label"].values
    logger.info(f"Label distribution: positive={int(labels.sum())}, negative={int((1 - labels).sum())}")

    # --- External features for ablation ---
    external_feats = _synthesize_external_features(meta_df)
    logger.info(f"External features shape: {external_feats.shape}")

    # --- Train/test split ---
    # Stratified split to maintain label balance
    X_emb_train, X_emb_test, y_train, y_test, meta_train, meta_test = train_test_split(
        embeddings, labels, meta_df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )
    ext_train = _synthesize_external_features(meta_train)
    ext_test = _synthesize_external_features(meta_test)

    logger.info(f"Train: {len(y_train)} ({int(y_train.sum())} pos) | Test: {len(y_test)} ({int(y_test.sum())} pos)")

    # Scale embeddings
    scaler_emb = StandardScaler()
    X_emb_train_s = scaler_emb.fit_transform(X_emb_train)
    X_emb_test_s = scaler_emb.transform(X_emb_test)

    scaler_ext = StandardScaler()
    ext_train_s = scaler_ext.fit_transform(ext_train)
    ext_test_s = scaler_ext.transform(ext_test)

    # Combined features
    X_combined_train = np.hstack([X_emb_train_s, ext_train_s])
    X_combined_test = np.hstack([X_emb_test_s, ext_test_s])

    # --- Define ablation configurations ---
    ablation_configs = {
        "embeddings_only": (X_emb_train_s, X_emb_test_s),
        "external_only": (ext_train_s, ext_test_s),
        "embeddings_plus_external": (X_combined_train, X_combined_test),
    }

    all_results = {}

    # --- Train models for each ablation ---
    for config_name, (X_tr, X_te) in ablation_configs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Ablation: {config_name} (features: {X_tr.shape[1]})")
        logger.info(f"{'='*60}")

        config_results = {}

        # Logistic regression (always run — fast baseline)
        logger.info("Training logistic regression ...")
        lr_metrics, lr_model = train_logistic(X_tr, y_train, X_te, y_test, seed=args.seed)
        config_results["logistic_regression"] = lr_metrics
        logger.info(f"  AUC-ROC: {lr_metrics.get('auc_roc', 0):.4f}")
        logger.info(f"  Avg Precision: {lr_metrics.get('average_precision', 0):.4f}")

        # XGBoost
        if not args.skip_xgboost:
            logger.info("Training XGBoost ...")
            xgb_metrics, xgb_model = train_xgboost(X_tr, y_train, X_te, y_test, seed=args.seed)
            if xgb_metrics:
                config_results["xgboost"] = xgb_metrics
                logger.info(f"  AUC-ROC: {xgb_metrics.get('auc_roc', 0):.4f}")
                logger.info(f"  Avg Precision: {xgb_metrics.get('average_precision', 0):.4f}")

        # MLP
        if not args.skip_mlp:
            logger.info("Training MLP ...")
            mlp_metrics, mlp_model = train_mlp(
                X_tr, y_train, X_te, y_test, seed=args.seed, epochs=args.epochs
            )
            config_results["mlp"] = mlp_metrics
            logger.info(f"  AUC-ROC: {mlp_metrics.get('auc_roc', 0):.4f}")
            logger.info(f"  Avg Precision: {mlp_metrics.get('average_precision', 0):.4f}")

        all_results[config_name] = config_results

    # --- Cross-validation (optional) ---
    if args.cv_folds > 0:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {args.cv_folds}-fold cross-validation (embeddings_only, XGBoost)")
        logger.info(f"{'='*60}")

        cv_aucs = []
        skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_emb_train_s, y_train)):
            X_fold_tr = X_emb_train_s[train_idx]
            y_fold_tr = y_train[train_idx]
            X_fold_val = X_emb_train_s[val_idx]
            y_fold_val = y_train[val_idx]

            fold_metrics, _ = train_xgboost(X_fold_tr, y_fold_tr, X_fold_val, y_fold_val, seed=args.seed)
            auc = fold_metrics.get("auc_roc", 0)
            cv_aucs.append(auc)
            logger.info(f"  Fold {fold+1}: AUC-ROC = {auc:.4f}")

        cv_results = {
            "folds": args.cv_folds,
            "auc_per_fold": cv_aucs,
            "auc_mean": float(np.mean(cv_aucs)),
            "auc_std": float(np.std(cv_aucs)),
        }
        all_results["cross_validation"] = cv_results
        logger.info(f"  CV AUC-ROC: {np.mean(cv_aucs):.4f} +/- {np.std(cv_aucs):.4f}")

    # --- Regional analysis ---
    logger.info("\n--- Regional AUC breakdown (embeddings_only, logistic regression) ---")
    regional_aucs = {}
    for region in meta_test["region"].unique():
        rmask = meta_test["region"].values == region
        if rmask.sum() < 10 or len(np.unique(y_test[rmask])) < 2:
            continue
        from sklearn.linear_model import LogisticRegression
        lr_tmp = LogisticRegression(C=1.0, max_iter=500, random_state=args.seed)
        lr_tmp.fit(X_emb_train_s, y_train)
        y_prob_r = lr_tmp.predict_proba(X_emb_test_s[rmask])[:, 1]
        try:
            auc_r = roc_auc_score(y_test[rmask], y_prob_r)
            regional_aucs[region] = float(auc_r)
            logger.info(f"  {region}: AUC = {auc_r:.4f} (n={rmask.sum()})")
        except ValueError:
            pass
    all_results["regional_auc"] = regional_aucs

    # --- Ablation summary table ---
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Config':<30} {'Model':<20} {'AUC-ROC':<10} {'Avg Prec':<10}")
    logger.info("-" * 70)
    for config_name, config_results in all_results.items():
        if config_name in ("cross_validation", "regional_auc"):
            continue
        for model_name, metrics in config_results.items():
            auc = metrics.get("auc_roc", 0)
            ap = metrics.get("average_precision", 0)
            logger.info(f"{config_name:<30} {model_name:<20} {auc:<10.4f} {ap:<10.4f}")

    # Compute ablation delta
    emb_auc = all_results.get("embeddings_only", {}).get("xgboost", {}).get("auc_roc")
    ext_auc = all_results.get("external_only", {}).get("xgboost", {}).get("auc_roc")
    combo_auc = all_results.get("embeddings_plus_external", {}).get("xgboost", {}).get("auc_roc")

    if emb_auc and ext_auc and combo_auc:
        logger.info(f"\nAblation delta (XGBoost):")
        logger.info(f"  OlmoEarth embeddings only:      AUC = {emb_auc:.4f}")
        logger.info(f"  External features only:          AUC = {ext_auc:.4f}")
        logger.info(f"  Embeddings + external:           AUC = {combo_auc:.4f}")
        logger.info(f"  Delta (combo - external):        +{combo_auc - ext_auc:.4f}")
        logger.info(f"  Delta (combo - embeddings):      +{combo_auc - emb_auc:.4f}")

        all_results["ablation_delta"] = {
            "embeddings_only_auc": emb_auc,
            "external_only_auc": ext_auc,
            "combined_auc": combo_auc,
            "delta_over_external": combo_auc - ext_auc,
            "delta_over_embeddings": combo_auc - emb_auc,
        }

    # --- Save results ---
    results_path = output_dir / "suitability_results.json"
    with open(str(results_path), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")

    # Save test predictions for further analysis
    if not args.skip_xgboost:
        try:
            import xgboost as xgb

            # Re-train best model on full embeddings+external for predictions
            best_model = xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", use_label_encoder=False,
                random_state=args.seed, n_jobs=-1,
            )
            best_model.fit(X_combined_train, y_train, verbose=False)
            y_prob_test = best_model.predict_proba(X_combined_test)[:, 1]

            predictions_df = meta_test.copy()
            predictions_df["y_true"] = y_test
            predictions_df["y_prob"] = y_prob_test
            predictions_df["y_pred"] = (y_prob_test >= 0.5).astype(int)
            predictions_path = output_dir / "test_predictions.csv"
            predictions_df.to_csv(str(predictions_path), index=False)
            logger.info(f"Test predictions saved to {predictions_path}")
        except Exception as e:
            logger.warning(f"Could not save test predictions: {e}")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
