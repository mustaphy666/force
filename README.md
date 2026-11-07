# Machine Learning-Based Lithology Prediction from Well Logs Using XGBoost

A machine learning project that uses conventional well-log measurements to predict subsurface lithology. The project applies feature engineering and an XGBoost classifier to the FORCE 2020 North Sea well-log dataset, evaluates performance using well-grouped cross-validation, and provides an interactive Streamlit application for lithology prediction from uploaded well-log files.

## Project Overview

Lithology identification is an important part of subsurface formation evaluation. Traditionally, geologists and petrophysicists interpret well logs alongside other geological information to identify the formations encountered in a borehole.

This project explores how machine learning can support that process by learning relationships between well-log measurements and labelled lithology intervals.

The model uses six conventional well-log curves:

- **GR:** Gamma Ray
- **RHOB:** Bulk Density
- **NPHI:** Neutron Porosity
- **DTC:** Compressional Sonic Transit Time
- **RDEP:** Deep Resistivity
- **PEF:** Photoelectric Factor

An XGBoost classifier is trained to predict lithology classes from these measurements and engineered features.

## Objectives

- Develop a machine learning workflow for lithology prediction from well logs.
- Apply geological and statistical feature engineering to conventional log measurements.
- Evaluate model performance using cross-validation grouped by well.
- Examine model performance across different lithology classes.
- Build an interactive application for uploading well-log data and viewing predictions.

## Dataset

The project uses the **FORCE 2020 Lithology Prediction dataset**, which contains labelled well-log data from the North Sea.

The dataset provides conventional well-log measurements and lithology labels from multiple wells. The model uses the available labelled observations to learn patterns associated with different lithologies.

**Target variable:** `FORCE_2020_LITHOFACIES_LITHOLOGY`

The model predicts ten lithology classes:

1. Sandstone
2. Shale
3. Sand/Shale
4. Limestone
5. Chalk
6. Dolomite
7. Marl
8. Anhydrite
9. Coal
10. Tuff

Dataset source: [FORCE 2020 Machine Learning Competition](https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition)

## Methodology

The project follows a structured machine learning workflow.

### 1. Data Preparation

- Load LAS well-log files.
- Standardise curve names and handle common naming variations.
- Replace recognised null-value sentinels.
- Retain labelled wells and the required log measurements.
- Sort observations by well and measured depth.
- Apply downsampling to reduce the number of training observations.

### 2. Feature Engineering

In addition to the original well-log measurements, the workflow generates features intended to capture local log behaviour and relationships between measurements.

These include:

- Log-transformed deep resistivity.
- Neutron-density difference feature (`NPHI_RHOB`).
- Rolling means and standard deviations using a 15-observation window.
- First-order differences for the input log curves and derived feature.

The resulting feature set contains 28 features.

### 3. Model Development

The classification model is built using **XGBoost**.

Key model parameters include:

| Parameter | Value |
|---|---:|
| Model | XGBClassifier |
| Number of estimators | 150 |
| Maximum depth | 6 |
| Learning rate | 0.1 |
| Tree method | Histogram |
| Random state | 42 |

The training workflow also applies class-balancing weights and analyst-defined weights based on the dataset's confidence codes.

### 4. Model Validation

The model is evaluated using **five-fold Stratified Group Cross-Validation**, with well identity used as the grouping variable.

This means observations from the same well are kept within the same fold rather than being randomly distributed across training and validation sets.

This provides a more realistic assessment of performance on wells excluded from the corresponding training fold.

Evaluation metrics include:

- Accuracy
- Macro F1-score
- Precision
- Recall
- Confusion matrix
- Per-class classification performance

## Model Performance

The current model produced the following out-of-fold results:

| Evaluation metric | Result |
|---|---:|
| Overall OOF accuracy | 69.36% |
| Overall pooled OOF macro F1 | 42.11% |
| Mean fold accuracy | 69.45% ± 2.95% |
| Mean fold macro F1 | 39.79% ± 3.38% |

The difference between accuracy and macro F1 is important. The dataset is imbalanced, with shale representing a substantial proportion of the observations. Consequently, overall accuracy does not fully reflect performance across less-represented lithologies.

The model performs relatively well on common classes such as shale and sandstone, while performance is weaker on some minority classes, including dolomite, chalk, and tuff.

These results should therefore be interpreted as a useful baseline rather than evidence of consistently high predictive performance across all lithologies.

## Feature Importance

The model's feature-importance analysis identifies Gamma Ray (`GR`) as the most influential feature, followed by the neutron-density difference feature (`NPHI_RHOB`).

Other influential features include neutron porosity, bulk density, and rolling statistics derived from the well logs.

Feature importance reflects how the trained model uses the available features; it does not establish a causal geological relationship.

## Interactive Application

A Streamlit application is included to make the trained model accessible through a simple interface.

The application allows users to:

- Upload LAS or CSV well-log files.
- Standardise common well-log curve names.
- Generate lithology predictions.
- View predicted lithology and confidence values.
- Visualise selected well-log curves alongside predicted lithology.
- Compare predictions with reference lithology labels when they are available.
- Download prediction results as a CSV file.

### Application Preview

*Add a screenshot of the running Streamlit application here.*

Suggested screenshot: the well-log tracks, predicted lithology column, and prediction summary displayed together.

## Project Structure

```text
lithology-prediction/
│
├── app/
│   └── app.py
│
├── src/
│   ├── features.py
│   └── train_final.py
│
├── model/
│   ├── xgboost_model.json
│   ├── metadata.json
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── data/
│   └── raw/
│       └── LAS well files
│
├── requirements.txt
├── README.md
└── .gitignore
```

*The filenames shown in the model directory are illustrative; retain the actual filenames generated by your training script.*

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Training

Place the labelled FORCE 2020 LAS files in the data directory expected by the training script.

Run the training pipeline with cross-validation enabled:

```bash
python src/train_final.py --cv
```

The pipeline performs data preparation, feature engineering, model training, cross-validation, and evaluation. It also saves the trained model and associated evaluation outputs.

*The training command assumes the script's configured data and output paths match the repository structure.*

## Running the Streamlit Application

After training, run:

```bash
streamlit run app/app.py
```

Upload a compatible LAS or CSV file through the application to generate lithology predictions.

For prediction, the application expects well-log measurements corresponding to the supported curves. Missing curves may be handled by the feature-generation workflow, although predictions will be less informative when important measurements are unavailable.

## Limitations

This project has several limitations:

- **Class imbalance:** Some lithology classes have substantially fewer observations than others, affecting their predictive performance.
- **Geographical scope:** The training data comes from the North Sea. The model has not been independently validated on wells from the Nigerian sedimentary basins or other geological settings.
- **Input dependence:** Prediction quality depends on the availability and quality of the input well-log measurements.
- **Confidence weighting:** The training workflow uses analyst-defined weights for the dataset's confidence codes. These weighting assumptions should not be interpreted as verified official confidence rankings.
- **Interpretation:** Machine learning predictions are intended to support geological interpretation, not replace geological and petrophysical expertise.

## Future Improvements

Potential extensions include:

- Testing the model on independent wells and datasets.
- Investigating improved approaches to minority lithology classification.
- Comparing XGBoost with other tree-based classification algorithms.
- Exploring geological constraints and additional petrophysical features.
- Evaluating model performance across different geological basins.

## Technologies Used

- Python
- NumPy
- pandas
- Matplotlib
- Seaborn
- scikit-learn
- XGBoost
- lasio
- Streamlit

## Author

**Mustapha Saheed**

Geology and Mineral Sciences graduate interested in applying machine learning and computational techniques to subsurface characterisation, petroleum geology, and geoscience problems.

- GitHub: [YOUR GITHUB PROFILE]
- LinkedIn: [YOUR LINKEDIN PROFILE]

---

*This project was developed as a portfolio demonstration of applied machine learning in geoscience, combining well-log data preparation, feature engineering, classification, model evaluation, and interactive visualisation.*
