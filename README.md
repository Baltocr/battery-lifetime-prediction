Early Prediction of Lithium-Ion Battery Lifetime Using Physics-Informed Machine Learning

This project uses the MIT-Stanford-Toyota battery dataset to predict lithium-ion battery cycle life from early-cycle electrochemical data.

The goal is to test whether features extracted from the first 100 cycles can predict long-term battery degradation, with special emphasis on physics-informed features derived from changes in the discharge capacity curve.

Project Motivation

Battery cycle-life testing can take months or years because cells often need to be cycled until they reach an end-of-life capacity threshold. Early lifetime prediction can help accelerate battery research, fast-charging optimization, and degradation analysis.

This project explores whether machine learning models can predict final cycle life using only early-cycle data.

Dataset

The dataset contains commercial lithium iron phosphate (LFP)/graphite cells cycled under fast-charging conditions. The raw data is provided as MATLAB .mat files containing per-cycle measurements such as discharge capacity, internal resistance, temperature, charge time, and voltage/capacity curves.

Raw data files are not included in this repository.

Current Features

The current feature table includes early-cycle summary features such as:

* discharge capacity at cycle 2 and cycle 100
* capacity fade from cycle 2 to cycle 100
* capacity retention at cycle 100
* internal resistance at cycle 2 and cycle 100
* internal resistance change from cycle 2 to cycle 100
* charge time at cycle 2
* average and maximum temperature at cycle 100

The project also includes Delta Q features computed from the difference between interpolated discharge capacity curves:

Delta Q = Qdlin(cycle 100) - Qdlin(cycle 10)

These features summarize how the discharge curve changes early in the battery’s lifetime.

Initial Model Results

Model	MAE cycles	RMSE cycles	R²
Dummy Mean Baseline	197.66	243.02	-0.084
Ridge Regression	57.94	77.71	0.889
Random Forest	66.80	108.31	0.785

Adding Delta Q features significantly improved model performance. Ridge Regression currently performs best, suggesting that physics-informed feature engineering can make simple models highly effective on this dataset.

Key Finding So Far

Random Forest feature importance shows that the most important predictors are Delta Q features, especially:

1. delta_q_var_100_10
2. delta_q_min_100_10
3. delta_q_mean_100_10

This supports the idea that early changes in discharge curve shape contain predictive information about long-term battery degradation.

Repository Structure

battery-lifetime-prediction/
├── app/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── reports/
│   └── figures/
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── models.py
│   └── visualization.py
├── README.md
├── requirements.txt
└── .gitignore

How to Run

Create and activate a virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Generate the feature table:

python src/data_loader.py

Train models and save results:

python src/models.py

Generate visualizations:

python src/visualization.py

Current Status

Completed:

* loaded all three raw structured dataset batches
* extracted valid cycle-life labels
* handled missing cycle-life values
* engineered early-cycle summary features
* engineered Delta Q features from cycles 10 and 100
* trained Dummy, Ridge Regression, and Random Forest models
* generated model comparison results
* generated predicted-vs-actual plot
* generated Delta Q curve visualization
* generated Random Forest feature importance plot

Next steps:

* add cross-validation
* improve model evaluation robustness
* add Ridge coefficient interpretation
* add more publication-quality figures
* build a Streamlit dashboard

