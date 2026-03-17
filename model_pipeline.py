import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class TwoStagePipeline:
    #########################
    # @author: Dorian (In case you have questions)
    # To give a quick explanaition about how I was thinking to deal with the zero inflated dataset
    # dataset is already cleaned and feature engineering was already performed.
    # The core idea is to create two-stage model, also called hurdle model.
    # 1) the first step is to create a classifier that predicts if a customer is going to spend money or not. intuitively we compute P(customer buys)
    #   input: X_cal, Y_cal
    #   output; y_hat = 1 if we predict that the customer is going to buy next season
    #                 = 0 if the classifier predicts that the customer will not spend any money next season
    # 2) we build a second model that trains only on the customer that were predicted as buyers.
    #   input: X_cal, Y_cal (only for observations were label is 1)
    #   output: y_hat_2 an estimator for the amount of money a buyer is going to spend.
    ###################################

    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parent

        self.features_path = self.BASE_DIR / "features.csv"
        self.train_path = self.BASE_DIR / "original_data/data/customer_clv_train.csv"
        self.test_path = self.BASE_DIR / "original_data/data/customer_clv_test.csv"

        self.date_cols = ["first_purchase", "last_purchase"]
        self.candidate_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

        self.features = None
        self.train = None
        self.test = None
        self.df = None
        self.X_test_full = None

        self.X = None
        self.y_class = None
        self.y_reg = None

        self.X_train = None
        self.X_val = None
        self.y_class_train = None
        self.y_class_val = None
        self.y_reg_train = None
        self.y_reg_val = None

        self.feature_columns = None

        self.clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        self.reg = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        self.best_threshold = None
        self.best_mae = float("inf")
        self.best_spearman = None
        self.best_pred_reg_val = None

    def load_data(self):
        self.features = pd.read_csv(self.features_path)
        self.train = pd.read_csv(self.train_path)
        self.test = pd.read_csv(self.test_path)

    def preprocess_dates(self):
        self.df = self.features.merge(self.train, on="cust_id", how="inner")

        # I need to handle the dates since i got some errors with sklearn and datetimes
        for col in self.date_cols:
            if col in self.features.columns:
                self.features[col] = pd.to_datetime(self.features[col], errors="coerce")
            if col in self.test.columns:
                self.test[col] = pd.to_datetime(self.test[col], errors="coerce")

        self.df = self.features.merge(self.train, on="cust_id", how="inner")
        self.X_test_full = self.features.merge(self.test, on="cust_id", how="right")

        for col in self.date_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].map(lambda x: x.toordinal() if pd.notnull(x) else np.nan)
            if col in self.X_test_full.columns:
                self.X_test_full[col] = self.X_test_full[col].map(lambda x: x.toordinal() if pd.notnull(x) else np.nan)

    def prepare_training_data(self):
        # defining the label for the first variable classification model
        self.df["target_binary"] = (self.df["revenue_2018_2019"] > 0).astype(int)
        self.df["target_reg"] = self.df["revenue_2018_2019"]

        self.X = self.df.drop(
            columns=["cust_id", "revenue_2018_2019", "target_binary", "target_reg"],
            errors="ignore"
        )
        self.X = self.X.select_dtypes(include=[np.number])
        self.X = self.X.fillna(0)

        self.y_class = self.df["target_binary"]
        self.y_reg = self.df["target_reg"]
        self.feature_columns = self.X.columns.tolist()

    def split_data(self):
        self.X_train, self.X_val, self.y_class_train, self.y_class_val, self.y_reg_train, self.y_reg_val = train_test_split(
            self.X,
            self.y_class,
            self.y_reg,
            test_size=0.2,
            random_state=42,
            stratify=self.y_class
        )

    def train_classifier(self):
        # The classifier, I choose for Random Forest, if predictions are bad we could try another model maybe
        self.clf.fit(self.X_train, self.y_class_train)

    def train_regressor(self):
        # Mechanism 2: Regressor model
        mask_train = self.y_class_train == 1

        # using the log transformation to obtain a lognormal model
        self.reg.fit(self.X_train[mask_train], np.log1p(self.y_reg_train[mask_train]))

    def tune_threshold(self):
        # I runned the model the first time with a treshold of 0.30, but i think we could obtain better results by finetuning the hyperparameters
        proba_val = self.clf.predict_proba(self.X_val)[:, 1]

        # Simple for-loop not very efficient but efficient enough for us since we are not graded on efficienty of the code
        for t in self.candidate_thresholds:
            pred_class_val = (proba_val >= t).astype(int)

            pred_reg_val_tmp = np.zeros(len(self.X_val))
            buyer_mask_pred = pred_class_val == 1

            if buyer_mask_pred.sum() > 0:
                pred_reg_val_tmp[buyer_mask_pred] = np.expm1(
                    self.reg.predict(self.X_val[buyer_mask_pred])
                )

            pred_reg_val_tmp = np.clip(pred_reg_val_tmp, 0, None)

            mae_t = mean_absolute_error(self.y_reg_val, pred_reg_val_tmp)
            spearman_t = spearmanr(self.y_reg_val, pred_reg_val_tmp).correlation

            print(f"Threshold={t:.2f} | MAE={mae_t:.4f} | Spearman={spearman_t:.4f}")

            if mae_t < self.best_mae:
                self.best_mae = mae_t
                self.best_threshold = t
                self.best_spearman = spearman_t
                self.best_pred_reg_val = pred_reg_val_tmp.copy()

        print("\nBest threshold found:")
        print(f"Threshold={self.best_threshold:.2f} | MAE={self.best_mae:.4f} | Spearman={self.best_spearman:.4f}")

    def evaluate(self):
        # To compute the final val metrics
        pred_reg_val = self.best_pred_reg_val

        mae = mean_absolute_error(self.y_reg_val, pred_reg_val)
        spearman_corr = spearmanr(self.y_reg_val, pred_reg_val).correlation

        print(f"\nValidation MAE: {mae:.4f}")
        print(f"Spearman Correlation: {spearman_corr:.4f}")

    def fit_full_data(self):
        # We also need to train the model on the full dataset
        self.clf.fit(self.X, self.y_class)

        mask_full = self.y_class == 1
        self.reg.fit(self.X[mask_full], np.log1p(self.y_reg[mask_full]))

    def prepare_test_data(self):
        # Once the model is trained we can do validate it on the test set
        X_test = self.X_test_full.drop(columns=["cust_id"], errors="ignore")
        X_test = X_test.select_dtypes(include=[np.number])
        X_test = X_test.reindex(columns=self.feature_columns, fill_value=0)
        X_test = X_test.fillna(0)

        return X_test

    def predict_test(self, X_test):
        proba_test = self.clf.predict_proba(X_test)[:, 1]
        pred_class_test = (proba_test >= self.best_threshold).astype(int)

        pred_test = np.zeros(len(X_test))
        buyer_mask_test = pred_class_test == 1

        if buyer_mask_test.sum() > 0:
            pred_test[buyer_mask_test] = np.expm1(self.reg.predict(X_test[buyer_mask_test]))

        pred_test = np.clip(pred_test, 0, None)
        return pred_test

    def save_submission(self, pred_test):
        submission = pd.DataFrame({
            "cust_id": self.X_test_full["cust_id"],
            "predicted_revenue_2018_2019": pred_test
        })

        submission.to_csv(self.BASE_DIR / "submission.csv", index=False)
        print("Submission saved!")

    def run(self):
        self.load_data()
        self.preprocess_dates()
        self.prepare_training_data()
        self.split_data()
        self.train_classifier()
        self.train_regressor()
        self.tune_threshold()
        self.evaluate()
        self.fit_full_data()
        X_test = self.prepare_test_data()
        pred_test = self.predict_test(X_test)
        self.save_submission(pred_test)


if __name__ == "__main__":
    pipeline = TwoStagePipeline()
    pipeline.run()