"""Hyperparameter tuning for the main LightGBM model via Optuna.

Searches learning rate, tree complexity (num_leaves, min_child_samples), row/column
sampling fractions, and L1/L2 regularization — optimizing validation PR-AUC with the
same time-based split and class-weighting approach that won the imbalance-handling
comparison (see reports/models/main_model_comparison.md). The best trial is retrained
and evaluated on the untouched test set; the tuned model only replaces the committed
one (models/main_model.txt) if it's genuinely better on test PR-AUC than the
hand-picked defaults, not just on the validation set Optuna optimized against.

Usage:
    .venv/Scripts/python.exe src/models/tune_hyperparameters.py
"""

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import optuna
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.load_data import load_merged  # noqa: E402
from models.features_tree import prepare_tree_features  # noqa: E402
from models.split import time_based_split  # noqa: E402
from models.train_main import best_threshold_f2, evaluate, pr_auc_feval  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports" / "models"
FIG_DIR = ROOT / "reports" / "figures"

# Test PR-AUC of the hand-picked-defaults model from main_model_comparison.md — the bar
# the tuned model has to clear on the held-out test set (not just the validation set
# Optuna optimizes against) to be adopted.
CURRENT_MODEL_TEST_PR_AUC = 0.5174

N_TRIALS = 30
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 50

COLOR_ACCENT = "#2a78d6"
COLOR_GRID = "#e1e0d9"


def objective(trial, X_train, y_train, X_val, y_val, categorical, scale_pos_weight):
    params = {
        "objective": "binary",
        "metric": "None",
        "boosting_type": "gbdt",
        "scale_pos_weight": scale_pos_weight,
        "seed": 42,
        "verbose": -1,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 200, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }

    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical, free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical, reference=train_set,
                           free_raw_data=False)
    booster = lgb.train(
        params, train_set,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[val_set],
        feval=pr_auc_feval,
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
    )
    val_proba = booster.predict(X_val, num_iteration=booster.best_iteration)
    trial.set_user_attr("best_iteration", booster.best_iteration)
    return average_precision_score(y_val, val_proba)


def plot_optimization_history(study: optuna.Study) -> None:
    values = [t.value for t in study.trials if t.value is not None]
    running_best = [max(values[:i + 1]) for i in range(len(values))]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(range(len(values)), values, color=COLOR_GRID, s=30, label="Trial value", zorder=2)
    ax.plot(range(len(running_best)), running_best, color=COLOR_ACCENT, linewidth=2,
            label="Best so far", zorder=3)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")
    ax.grid(color=COLOR_GRID, linewidth=0.8)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Validation PR-AUC")
    ax.set_title("Optuna optimization history", loc="left", fontsize=11)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hpo_optimization_history.png", dpi=150)
    plt.close(fig)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("Loading merged train data...")
    df = load_merged("train")
    train, val, test = time_based_split(df)

    train, numeric, categorical = prepare_tree_features(train)
    val, _, _ = prepare_tree_features(val)
    test, _, _ = prepare_tree_features(test)
    feature_cols = numeric + categorical

    X_train, y_train = train[feature_cols], train["isFraud"]
    X_val, y_val = val[feature_cols], val["isFraud"]
    X_test, y_test = test[feature_cols], test["isFraud"]
    spw = (y_train == 0).sum() / (y_train == 1).sum()

    print(f"Running Optuna search ({N_TRIALS} trials, optimizing validation PR-AUC)...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    t0 = time.time()

    def _obj(trial):
        return objective(trial, X_train, y_train, X_val, y_val, categorical, spw)

    study.optimize(_obj, n_trials=N_TRIALS, show_progress_bar=False)
    print(f"Search finished in {time.time() - t0:.0f}s")
    print(f"Best validation PR-AUC: {study.best_value:.4f}")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")

    plot_optimization_history(study)

    print("\nRetraining best params on train, evaluating on held-out test...")
    best_params = {
        "objective": "binary", "metric": "None", "boosting_type": "gbdt",
        "scale_pos_weight": spw, "seed": 42, "verbose": -1,
        **study.best_params,
    }
    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical, free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical, reference=train_set,
                           free_raw_data=False)
    tuned_booster = lgb.train(
        best_params, train_set, num_boost_round=NUM_BOOST_ROUND, valid_sets=[val_set],
        feval=pr_auc_feval,
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
    )

    proba_val = tuned_booster.predict(X_val, num_iteration=tuned_booster.best_iteration)
    tuned_threshold, _ = best_threshold_f2(y_val, proba_val)
    proba_test = tuned_booster.predict(X_test, num_iteration=tuned_booster.best_iteration)
    tuned_results = evaluate(y_test, proba_test, tuned_threshold)

    print(f"\nTuned model test PR-AUC: {tuned_results['pr_auc']:.4f} "
          f"(current committed model: {CURRENT_MODEL_TEST_PR_AUC:.4f})")

    adopted = tuned_results["pr_auc"] > CURRENT_MODEL_TEST_PR_AUC
    if adopted:
        print("Tuned model is better on test — adopting it as the new main model.")
        final_model_path = MODELS_DIR / "main_model.txt"
        tuned_booster.save_model(str(final_model_path), num_iteration=tuned_booster.best_iteration)
        meta = {
            "numeric_features": numeric,
            "categorical_features": categorical,
            "threshold": float(tuned_threshold),
            "scale_pos_weight": float(spw),
            "best_iteration": tuned_booster.best_iteration,
            "hyperparameters": study.best_params,
        }
        (MODELS_DIR / "main_model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Saved to {final_model_path}")
    else:
        print("Tuned model did NOT beat the current committed model on test PR-AUC — "
              "keeping the existing model, documenting the search anyway.")

    report_lines = [
        "# Hyperparameter Tuning — Optuna",
        "",
        f"Searched {N_TRIALS} trials over learning rate, tree complexity "
        "(`num_leaves`, `min_child_samples`), row/column sampling fractions, and L1/L2 "
        "regularization, optimizing validation PR-AUC with the same time-based split "
        "and class-weighting approach as "
        "[main_model_comparison.md](main_model_comparison.md). TPE sampler (Optuna's "
        "default), seeded for reproducibility.",
        "",
        "![Optimization history](../figures/hpo_optimization_history.png)",
        "",
        "## Best trial",
        "",
        f"Validation PR-AUC: **{study.best_value:.4f}**",
        "",
        "| Hyperparameter | Value |",
        "|---|---|",
    ]
    for k, v in study.best_params.items():
        v_str = f"{v:.4g}" if isinstance(v, float) else str(v)
        report_lines.append(f"| `{k}` | {v_str} |")

    report_lines += [
        "",
        "## Test set result",
        "",
        "| Model | Test PR-AUC | Test ROC-AUC |",
        "|---|---|---|",
        f"| Hand-picked defaults (current) | {CURRENT_MODEL_TEST_PR_AUC:.4f} | 0.8929 |",
        f"| Optuna-tuned | {tuned_results['pr_auc']:.4f} | {tuned_results['roc_auc']:.4f} |",
        "",
    ]
    if adopted:
        report_lines += [
            f"**Adopted.** The tuned model improved test PR-AUC from "
            f"{CURRENT_MODEL_TEST_PR_AUC:.4f} to {tuned_results['pr_auc']:.4f} "
            f"({(tuned_results['pr_auc'] / CURRENT_MODEL_TEST_PR_AUC - 1):+.1%} relative) "
            "and now backs the API/dashboard (`models/main_model.txt`).",
            "",
        ]
    else:
        report_lines += [
            f"**Not adopted.** The tuned model's test PR-AUC ({tuned_results['pr_auc']:.4f}) "
            f"did not exceed the hand-picked defaults' ({CURRENT_MODEL_TEST_PR_AUC:.4f}) — "
            "the committed model is unchanged. This is itself a useful result: it suggests "
            "the original parameter choices (moderate learning rate, num_leaves=63, "
            "0.8 sampling fractions) were already close to a local optimum for this "
            "problem, and the earlier imbalance-handling technique and threshold "
            "tuning contributed far more to the final PR-AUC than further hyperparameter "
            "search does on top of them.",
            "",
        ]
    (REPORT_DIR / "hyperparameter_tuning.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nReport saved to {REPORT_DIR / 'hyperparameter_tuning.md'}")


if __name__ == "__main__":
    main()
