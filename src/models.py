import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_feature_table(path="data/processed/summary_features.csv"):
    """
    Load the early-cycle summary feature table.
    """
    return pd.read_csv(path)

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained regression model.
    Returns predictions, MAE, RMSE, and R².
    """
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    return predictions, mae, rmse, r2

def cross_validate_model(model, X, y, cv):
    """
    Run cross-validation for a regression model.
    Returns mean and standard deviation for MAE, RMSE, and R².
    """
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        }
    )

    results = {
        "mae_mean": -scores["test_mae"].mean(),
        "mae_std": scores["test_mae"].std(),
        "rmse_mean": -scores["test_rmse"].mean(),
        "rmse_std": scores["test_rmse"].std(),
        "r2_mean": scores["test_r2"].mean(),
        "r2_std": scores["test_r2"].std(),
    }

    return results

if __name__ == "__main__":
    df = load_feature_table()

    target_col = "cycle_life"

    non_feature_cols = ["batch_file", "cell_index", target_col]
    feature_cols = [col for col in df.columns if col not in non_feature_cols]

    X = df[feature_cols]
    y = df[target_col]

    print("Feature columns:")
    print(feature_cols)

    print("\nX shape:")
    print(X.shape)

    print("\ny shape:")
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = DummyRegressor(strategy="mean")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\nDummy baseline results:")
    print(f"MAE: {mae:.2f} cycles")
    print(f"RMSE: {rmse:.2f} cycles")
    print(f"R²: {r2:.3f}")

    ridge_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    ridge_cv_scores = cross_validate(
        ridge_model,
        X,
        y,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        }
    )

    print("\nRidge regression 5-fold cross-validation:")
    print(f"MAE: {-ridge_cv_scores['test_mae'].mean():.2f} ± {ridge_cv_scores['test_mae'].std():.2f} cycles")
    print(f"RMSE: {-ridge_cv_scores['test_rmse'].mean():.2f} ± {ridge_cv_scores['test_rmse'].std():.2f} cycles")
    print(f"R²: {ridge_cv_scores['test_r2'].mean():.3f} ± {ridge_cv_scores['test_r2'].std():.3f}")
    ridge_model.fit(X_train, y_train)

    ridge_predictions, ridge_mae, ridge_rmse, ridge_r2 = evaluate_model(
    ridge_model,
    X_test,
    y_test
)

    print("\nRidge regression results:")
    print(f"MAE: {ridge_mae:.2f} cycles")
    print(f"RMSE: {ridge_rmse:.2f} cycles")
    print(f"R²: {ridge_r2:.3f}")

    rf_model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=3
    )

    rf_cv_scores = cross_validate(
        rf_model,
        X,
        y,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        }
    )

    print("\nRandom Forest 5-fold cross-validation:")
    print(f"MAE: {-rf_cv_scores['test_mae'].mean():.2f} ± {rf_cv_scores['test_mae'].std():.2f} cycles")
    print(f"RMSE: {-rf_cv_scores['test_rmse'].mean():.2f} ± {rf_cv_scores['test_rmse'].std():.2f} cycles")
    print(f"R²: {rf_cv_scores['test_r2'].mean():.3f} ± {rf_cv_scores['test_r2'].std():.3f}")

    rf_model.fit(X_train, y_train)

    rf_predictions, rf_mae, rf_rmse, rf_r2 = evaluate_model(
        rf_model,
        X_test,
        y_test
    )

    print("\nRandom Forest results:")
    print(f"MAE: {rf_mae:.2f} cycles")
    print(f"RMSE: {rf_rmse:.2f} cycles")
    print(f"R²: {rf_r2:.3f}")

    results = pd.DataFrame([
        {
            "model": "Dummy Mean Baseline",
            "evaluation": "holdout",
            "mae_cycles": mae,
            "rmse_cycles": rmse,
            "r2": r2,
        },
        {
            "model": "Ridge Regression",
            "evaluation": "holdout",
            "mae_cycles": ridge_mae,
            "rmse_cycles": ridge_rmse,
            "r2": ridge_r2,
        },
        {
            "model": "Random Forest",
            "evaluation": "holdout",
            "mae_cycles": rf_mae,
            "rmse_cycles": rf_rmse,
            "r2": rf_r2,
        },
        {
            "model": "Ridge Regression",
            "evaluation": "5-fold CV mean",
            "mae_cycles": -ridge_cv_scores["test_mae"].mean(),
            "rmse_cycles": -ridge_cv_scores["test_rmse"].mean(),
            "r2": ridge_cv_scores["test_r2"].mean(),
        },
        {
            "model": "Random Forest",
            "evaluation": "5-fold CV mean",
            "mae_cycles": -rf_cv_scores["test_mae"].mean(),
            "rmse_cycles": -rf_cv_scores["test_rmse"].mean(),
            "r2": rf_cv_scores["test_r2"].mean(),
        },
    ])

    importances = rf_model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print("\nRandom Forest feature importances:")
    print(importance_df)

    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    importance_output_path = "reports/feature_importances.csv"
    importance_df.to_csv(importance_output_path, index=False)
    print(f"\nSaved feature importances to: {importance_output_path}")

    plt.figure(figsize=(8, 6))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.gca().invert_yaxis()

    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.tight_layout()

    importance_plot_path = figures_dir / "random_forest_feature_importance.png"
    plt.savefig(importance_plot_path, dpi=300)
    plt.close()

    print(f"Saved feature importance plot to: {importance_plot_path}")

    output_path = "reports/model_results.csv"
    results.to_csv(output_path, index=False)

    print(f"\nSaved model results to: {output_path}")
    print("\nModel comparison:")
    print(results)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, rf_predictions, alpha=0.8)

    min_value = min(y_test.min(), rf_predictions.min())
    max_value = max(y_test.max(), rf_predictions.max())

    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

    plt.scatter(y_test, ridge_predictions, alpha=0.8)

    min_value = min(y_test.min(), ridge_predictions.min())
    max_value = max(y_test.max(), ridge_predictions.max())

    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

    plt.title("Ridge Regression: Predicted vs Actual Cycle Life")
    plt.xlabel("Actual Cycle Life")
    plt.ylabel("Predicted Cycle Life")

    plt.tight_layout()

    output_plot = figures_dir / "ridge_predicted_vs_actual.png"
    plt.savefig(output_plot, dpi=300)
    plt.close()

    print(f"\nSaved predicted vs actual plot to: {output_plot}")