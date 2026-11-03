# Online Shopper Purchase Prediction

## a. Problem statement
 Predict whether an e-commerce browsing session ends in a purchase (`Revenue`) from session-level behaviour (pages visited, time spent, bounce/exit rates, page value, month, visitor type). Compare five classifiers on the same data using Accuracy, AUC, Precision, Recall, F1 and MCC, and say why this matters (targeting buyers, spotting sessions at risk of abandoning).

## b. Dataset description
- **Source:** Online Shoppers Purchasing Intention Dataset, UCI Machine Learning Repository (id 468): https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- **Instances:** 12,330 sessions (the notebook drops exact duplicate rows; fill in the final count from its output)
- **Features:** 17 (10 numeric, 7 categorical/integer-coded): `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`, `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`
- **Target:** `Revenue` (binary: True = session ended in a purchase)
- **Class balance:** about 15% purchases, so imbalanced
- **Split:** 80/20 stratified, `random_state=42`. The 20% test split is saved as `test_data.csv`.
- **Preprocessing:** median imputation + standard scaling for numeric columns, one-hot encoding for `Month`, `VisitorType` and `Weekend`, all inside each saved pipeline. Class weights are balanced for Logistic Regression, Decision Tree and SVM.

## c. GitHub repository
[Git hub](https://github.com/ManobharathAI/-Online-Shopper-Purchase-Prediction)

[Live Stream Lit app](https://gtwckxfpl6ezqmtab85f8c.streamlit.app/)

## d. Models used

Metrics on the held-out test split (20%, stratified, `random_state=42`):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8513 | 0.9109 | 0.5164 | 0.7853 | 0.6231 | 0.5536 |
| Decision Tree | 0.8439 | 0.9220 | 0.5008 | 0.8246 | 0.6231 | 0.5583 |
| KNN | 0.8791 | 0.8355 | 0.7231 | 0.3691 | 0.4887 | 0.4595 |
| Naive Bayes | 0.7108 | 0.8263 | 0.3288 | 0.8141 | 0.4684 | 0.3772 |
| Support Vector Machine | 0.8681 | 0.9064 | 0.5573 | 0.7644 | 0.6446 | 0.5767 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression gave an accuracy of 0.8513 and an AUC of 0.9109. Its recall of 0.7853 means it found most of the buyers, but its precision of 0.5164 shows that about half of the sessions it flagged as purchases were not. Since it is a simple linear model, the AUC shows that the features already separate buyers and non-buyers quite well. Its MCC of 0.5536 is decent but lower than SVM. |
| Decision Tree | The Decision Tree had the best AUC (0.9220) and the best recall (0.8246), so it was the best at catching actual buyers. Its precision was the lowest of the stronger models (0.5008), which means many false alarms. Its accuracy of 0.8439 looks fine, but it is close to what we get by always predicting "no purchase", so accuracy alone is misleading here. I limited the depth to avoid overfitting. |
| KNN | KNN had the highest accuracy (0.8791) and the highest precision (0.7231), but its recall was only 0.3691, so it missed about two thirds of the real buyers. Its F1 (0.4887) and MCC (0.4595) are therefore low. This happens because most sessions are non-purchases, so the nearest neighbours are mostly the majority class. KNN has no class weight option, which also made it worse than the other models. |
| Naive Bayes | Naive Bayes was the weakest model overall, with accuracy 0.7108, precision 0.3288 and MCC 0.3772. It has a high recall (0.8141) but only because it predicts "purchase" too often, which gives many false positives. This is probably because it assumes the features are independent, but features like page counts and durations are strongly correlated. |
| Support Vector Machine | SVM gave the best balance between precision (0.5573) and recall (0.7644), so it got the highest F1 (0.6446) and the highest MCC (0.5767). Its AUC (0.9064) is close to the best. The RBF kernel can learn non-linear patterns in the sessions, which Logistic Regression cannot, and class weighting helped it detect the minority purchase class. |
| **Overall winner for this dataset** | **Support Vector Machine.** The dataset is imbalanced (only about 15% of sessions end in a purchase), so I judged the models on MCC and F1 instead of accuracy. SVM had the highest of both. KNN has the best accuracy, but it misses most buyers, so it is not the best choice. If the goal was only to catch as many buyers as possible, the Decision Tree would be a good alternative because of its higher recall and AUC. |

## Project structure
```
app.py                  Streamlit app (upload CSV, pick model, metrics, confusion matrix, report)
common.py               metric computation shared by notebook and app
Source.ipynb   downloads data, trains and saves the five models, writes test_data.csv
requirements.txt
test_data.csv           held-out test split (generated by the notebook)
model/                  *.joblib pipelines, meta.json, metrics.csv (generated by the notebook)
```

## How to run
```
pip install -r requirements.txt jupyter
# open Source.ipynb and Run All
streamlit run app.py
```
