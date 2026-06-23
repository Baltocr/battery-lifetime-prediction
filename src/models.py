import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_feature_table(path="data/processed/summary_features.csv"):
    """
    Load the early-cycle summary feature table.
    """
    return pd.read_csv(path)


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

    ridge_model.fit(X_train, y_train)

    ridge_predictions = ridge_model.predict(X_test)

    ridge_mae = mean_absolute_error(y_test, ridge_predictions)
    ridge_rmse = mean_squared_error(y_test, ridge_predictions) ** 0.5
    ridge_r2 = r2_score(y_test, ridge_predictions)

    print("\nRidge regression results:")
    print(f"MAE: {ridge_mae:.2f} cycles")
    print(f"RMSE: {ridge_rmse:.2f} cycles")
    print(f"R²: {ridge_r2:.3f}")

    rf_model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=3
    )

    rf_model.fit(X_train, y_train)

    rf_predictions = rf_model.predict(X_test)

    rf_mae = mean_absolute_error(y_test, rf_predictions)
    rf_rmse = mean_squared_error(y_test, rf_predictions) ** 0.5
    rf_r2 = r2_score(y_test, rf_predictions)

    print("\nRandom Forest results:")
    print(f"MAE: {rf_mae:.2f} cycles")
    print(f"RMSE: {rf_rmse:.2f} cycles")
    print(f"R²: {rf_r2:.3f}")

    results = pd.DataFrame([
        {
            "model": "Dummy Mean Baseline",
            "mae_cycles": mae,
            "rmse_cycles": rmse,
            "r2": r2,
        },
        {
            "model": "Ridge Regression",
            "mae_cycles": ridge_mae,
            "rmse_cycles": ridge_rmse,
            "r2": ridge_r2,
        },
        {
            "model": "Random Forest",
            "mae_cycles": rf_mae,
            "rmse_cycles": rf_rmse,
            "r2": rf_r2,
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