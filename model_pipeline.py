import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class TwoStagePipeline:
    """
    @author: Dorian
    Each instance of this class represents the state of a two staged model.
    The pipeline is set up in such a way that if you want to change the model you just need to modify it in the constructor (__init__ function)
    If needed you can add a new function, but try to keep a coherent structure.
    Every already existing function can be modified but leave a comment when you modify please.
    LEGIT functions shouldn't be modified normally.

    Two-stage CLV pipeline

    Stage 1:
        Classifier predicts probability that customer will buy something in the next season. So this classifier only focuses on classying customers as 0 and 1

    Stage 2:
        Regressor that only works on the customers classified as 1 to predicts how much they are going to spend.

    Final prediction:
        P(buy) * E(revenue | buy)

    """

    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parent

        self.features_path = self.BASE_DIR / "data" / "features.csv"
        self.train_path = self.BASE_DIR / "data" / "original_data" / "data" / "customer_clv_train.csv"
        self.test_path = self.BASE_DIR / "data" / "original_data" / "data" / "customer_clv_test.csv"

        self.date_cols = ["first_purchase", "last_purchase"]

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
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        self.reg = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )

        self.val_pred_expected = None
        self.val_pred_threshold = None
        self.best_threshold = None
        self.best_threshold_mae = float("inf")
        self.best_threshold_spearman = None
    
    #LEGIT
    def load_data(self):
        self.features = pd.read_csv(self.features_path)
        self.train = pd.read_csv(self.train_path)
        self.test = pd.read_csv(self.test_path)

        print(f"Features shape: {self.features.shape}")
        print(f"Train shape: {self.train.shape}")
        print(f"Test shape: {self.test.shape}")

    def preprocess_dates(self):
        for col in self.date_cols:
            if col in self.features.columns:
                self.features[col] = pd.to_datetime(self.features[col], errors="coerce")

        self.df = self.features.merge(self.train, on="cust_id", how="inner")
        self.X_test_full = self.features.merge(self.test, on="cust_id", how="right")

        for col in self.date_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].map(
                    lambda x: x.toordinal() if pd.notnull(x) else np.nan
                )
            if col in self.X_test_full.columns:
                self.X_test_full[col] = self.X_test_full[col].map(
                    lambda x: x.toordinal() if pd.notnull(x) else np.nan
                )
    
    #LEGIT
    def prepare_training_data(self):
        self.df["target_binary"] = (self.df["revenue_2018_2019"] > 0).astype(int)
        self.df["target_reg"] = self.df["revenue_2018_2019"]

        self.X = self.df.drop(
            columns=["cust_id", "revenue_2018_2019", "target_binary", "target_reg"],
            errors="ignore"
        )

        self.X = self.X.select_dtypes(include=[np.number]).copy()
        self.X = self.X.fillna(0)

        self.y_class = self.df["target_binary"]
        self.y_reg = self.df["target_reg"]

        self.feature_columns = self.X.columns.tolist()

        print(f"Training matrix shape: {self.X.shape}")
        print(f"Positive buyers ratio: {self.y_class.mean():.4f}")
    
    #LEGIT
    def split_data(self):
        (
            self.X_train,
            self.X_val,
            self.y_class_train,
            self.y_class_val,
            self.y_reg_train,
            self.y_reg_val
        ) = train_test_split(
            self.X,
            self.y_class,
            self.y_reg,
            test_size=0.2,
            random_state=42,
            stratify=self.y_class
        )

        print(f"X_train shape: {self.X_train.shape}")
        print(f"X_val shape: {self.X_val.shape}")

    def train_classifier(self):
        self.clf.fit(self.X_train, self.y_class_train)

    def train_regressor(self):
        mask_train = self.y_class_train == 1

        if mask_train.sum() == 0:
            raise ValueError("No positive buyers in training set for regressor.")

        self.reg.fit(
            self.X_train[mask_train],
            np.log1p(self.y_reg_train[mask_train])
        )

    def _safe_spearman(self, y_true, y_pred):
        corr = spearmanr(y_true, y_pred).correlation
        return 0.0 if pd.isna(corr) else corr

    def evaluate_probability_weighted(self):
        proba_val = self.clf.predict_proba(self.X_val)[:, 1]

        auc = roc_auc_score(self.y_class_val, proba_val)
        print(f"\nClassifier AUC: {auc:.4f}")
        print(f"Validation probability stats:")
        print(f"  min  = {proba_val.min():.4f}")
        print(f"  max  = {proba_val.max():.4f}")
        print(f"  mean = {proba_val.mean():.4f}")

        reg_pred_log = self.reg.predict(self.X_val)
        reg_pred = np.expm1(reg_pred_log)
        reg_pred = np.clip(reg_pred, 0, None)

        self.val_pred_expected = proba_val * reg_pred
        self.val_pred_expected = np.clip(self.val_pred_expected, 0, None)

        mae = mean_absolute_error(self.y_reg_val, self.val_pred_expected)
        spearman_corr = self._safe_spearman(self.y_reg_val, self.val_pred_expected)

        print("\nProbability-weighted validation results:")
        print(f"MAE: {mae:.4f}")
        print(f"Spearman: {spearman_corr:.4f}")

    def evaluate_threshold_versions(self):
        candidate_thresholds = np.arange(0.05, 0.55, 0.05)
        proba_val = self.clf.predict_proba(self.X_val)[:, 1]

        for t in candidate_thresholds:
            pred_class_val = (proba_val >= t).astype(int)

            pred_reg_val_tmp = np.zeros(len(self.X_val))
            buyer_mask_pred = pred_class_val == 1

            if buyer_mask_pred.sum() > 0:
                pred_reg_val_tmp[buyer_mask_pred] = np.expm1(
                    self.reg.predict(self.X_val[buyer_mask_pred])
                )

            pred_reg_val_tmp = np.clip(pred_reg_val_tmp, 0, None)

            mae_t = mean_absolute_error(self.y_reg_val, pred_reg_val_tmp)
            spearman_t = self._safe_spearman(self.y_reg_val, pred_reg_val_tmp)

            print(
                f"Threshold={t:.2f} | "
                f"pred_buyers={buyer_mask_pred.sum()} | "
                f"MAE={mae_t:.4f} | "
                f"Spearman={spearman_t:.4f}"
            )

            if mae_t < self.best_threshold_mae:
                self.best_threshold_mae = mae_t
                self.best_threshold = t
                self.best_threshold_spearman = spearman_t
                self.val_pred_threshold = pred_reg_val_tmp.copy()

        print("\nBest hard-threshold result:")
        print(
            f"Threshold={self.best_threshold:.2f} | "
            f"MAE={self.best_threshold_mae:.4f} | "
            f"Spearman={self.best_threshold_spearman:.4f}"
        )

    def compare_validation_strategies(self):
        mae_expected = mean_absolute_error(self.y_reg_val, self.val_pred_expected)
        sp_expected = self._safe_spearman(self.y_reg_val, self.val_pred_expected)

        mae_thresh = mean_absolute_error(self.y_reg_val, self.val_pred_threshold)
        sp_thresh = self._safe_spearman(self.y_reg_val, self.val_pred_threshold)

        print("\nComparison of prediction strategies:")
        print(f"Expected value approach  -> MAE={mae_expected:.4f}, Spearman={sp_expected:.4f}")
        print(f"Hard threshold approach  -> MAE={mae_thresh:.4f}, Spearman={sp_thresh:.4f}")

        if mae_expected <= mae_thresh:
            print("Chosen final strategy: probability-weighted expected revenue")
            return "expected"
        else:
            print("Chosen final strategy: hard threshold")
            return "threshold"

    def fit_full_data(self):
        self.clf.fit(self.X, self.y_class)

        mask_full = self.y_class == 1
        if mask_full.sum() == 0:
            raise ValueError("No positive buyers in full training set for regressor.")

        self.reg.fit(
            self.X[mask_full],
            np.log1p(self.y_reg[mask_full])
        )

    #LEGIT
    def prepare_test_data(self):
        X_test = self.X_test_full.drop(columns=["cust_id"], errors="ignore")
        X_test = X_test.select_dtypes(include=[np.number]).copy()
        X_test = X_test.reindex(columns=self.feature_columns, fill_value=0)
        X_test = X_test.fillna(0)

        missing_feature_rows = X_test.isna().all(axis=1).sum()
        print(f"Completely empty test feature rows: {missing_feature_rows}")

        return X_test

    def predict_test_expected(self, X_test):
        proba_test = self.clf.predict_proba(X_test)[:, 1]
        reg_pred = np.expm1(self.reg.predict(X_test))
        reg_pred = np.clip(reg_pred, 0, None)

        pred_test = proba_test * reg_pred
        pred_test = np.clip(pred_test, 0, None)

        print(f"Predictions summary (expected value):")
        print(f"  min  = {pred_test.min():.4f}")
        print(f"  max  = {pred_test.max():.4f}")
        print(f"  mean = {pred_test.mean():.4f}")

        return pred_test

    def predict_test_threshold(self, X_test):
        proba_test = self.clf.predict_proba(X_test)[:, 1]
        pred_class_test = (proba_test >= self.best_threshold).astype(int)

        pred_test = np.zeros(len(X_test))
        buyer_mask_test = pred_class_test == 1

        print("Number of predicted buyers (test):", buyer_mask_test.sum())

        if buyer_mask_test.sum() > 0:
            pred_test[buyer_mask_test] = np.expm1(
                self.reg.predict(X_test[buyer_mask_test])
            )

        pred_test = np.clip(pred_test, 0, None)

        print(f"Predictions summary (threshold):")
        print(f"  min  = {pred_test.min():.4f}")
        print(f"  max  = {pred_test.max():.4f}")
        print(f"  mean = {pred_test.mean():.4f}")

        return pred_test

    #LEGIT
    def save_submission(self, pred_test, filename="submission.csv"):
        submission = pd.DataFrame({
            "cust_id": self.X_test_full["cust_id"],
            "predicted_revenue_2018_2019": pred_test
        })

        output_path = self.BASE_DIR / filename
        submission.to_csv(output_path, index=False)
        print(f"Submission saved to: {output_path}")

    def run(self):
        self.load_data()
        self.preprocess_dates()
        self.prepare_training_data()
        self.split_data()

        self.train_classifier()
        self.train_regressor()

        self.evaluate_probability_weighted()
        self.evaluate_threshold_versions()
        chosen_strategy = self.compare_validation_strategies()

        self.fit_full_data()
        X_test = self.prepare_test_data()

        if chosen_strategy == "expected":
            pred_test = self.predict_test_expected(X_test)
            self.save_submission(pred_test, filename="submission_expected_value.csv")
        else:
            pred_test = self.predict_test_threshold(X_test)
            self.save_submission(pred_test, filename="submission_threshold.csv")


if __name__ == "__main__":
    pipeline = TwoStagePipeline()
    pipeline.run()