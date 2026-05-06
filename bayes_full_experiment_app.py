# -*- coding: utf-8 -*-
"""
Demo thực nghiệm và đánh giá kết quả
Đề tài: Thuật toán phân loại Bayes ứng dụng cho bài toán phân loại cảm xúc văn bản

Bản này chỉ giữ các chức năng chạy demo/thực nghiệm:
- Tải dữ liệu CSV.
- Xem thống kê và xem trước dữ liệu.
- Chạy GridSearchCV + 5-fold Stratified Cross Validation.
- Chia train/test và đánh giá trên tập kiểm tra.
- So sánh MultinomialNB, BernoulliNB, GaussianNB và Logistic Regression baseline.
- Hiển thị Accuracy, Precision, Recall, F1-score, ROC-AUC.
- Hiển thị Confusion Matrix, Classification Report và các mẫu dự đoán sai.
- Dự đoán thử văn bản mới.
- Giải thích từ ảnh hưởng tới dự đoán.
- Lưu mô hình tốt nhất.
- Tải lại mô hình đã lưu.
- Xuất bảng kết quả thực nghiệm ra CSV.

Lưu ý: Bản demo này chỉ giả định dữ liệu tiếng Anh trong file CSV.
Nếu dữ liệu là ngôn ngữ khác, kết quả nhận dạng và đánh giá có thể không chính xác.

Cách chạy:
    pip install -r requirements.txt
    python bayes_full_experiment_app.py

File dữ liệu mặc định đặt cùng thư mục:
    du_lieu_nhan_xet_phim_bayes.csv
"""

from __future__ import annotations

import json
import re
import traceback
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    import joblib
    import numpy as np
    import pandas as pd

    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import (
        train_test_split,
        StratifiedKFold,
        GridSearchCV,
        cross_validate,
    )
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
        classification_report,
    )
except Exception as e:
    IMPORT_ERROR = str(e)
else:
    IMPORT_ERROR = ""


# ============================================================
# CẤU HÌNH
# ============================================================

DEFAULT_CSV = "du_lieu_nhan_xet_phim_bayes.csv"
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CV_FOLDS = 5

NEGATIVE_LABEL = "Âm tính"
POSITIVE_LABEL = "Dương tính"
LABEL_ORDER = [NEGATIVE_LABEL, POSITIVE_LABEL]

# Một ít stopword tiếng Việt để có lựa chọn trong GridSearchCV.
# Grid vẫn có cả trường hợp stop_words=None để khớp thực nghiệm trong báo cáo.
VIETNAMESE_STOPWORDS = [
    "và", "là", "của", "có", "cho", "trong", "với", "một", "những", "các",
    "này", "đó", "rất", "tôi", "thấy", "được", "bị", "đã", "sẽ", "thì",
    "khi", "vì", "nên", "cũng", "khá", "nhiều", "ít", "hơn", "để", "từ",
]

REPORT_REFERENCE_RESULTS = {
    "MultinomialNB + CountVectorizer": {
        "Mục đích": "Mô hình Bayes chính cho dữ liệu văn bản dạng tần suất từ.",
    },
    "MultinomialNB + TfidfVectorizer": {
        "Mục đích": "Biến thể dùng trọng số TF-IDF để giảm ảnh hưởng của từ phổ biến.",
    },
    "BernoulliNB + Binary CountVectorizer": {
        "Mục đích": "Mô hình Bayes nhị phân, quan tâm từ có xuất hiện hay không.",
    },
    "GaussianNB + TfidfVectorizer": {
        "Mục đích": "Mô hình Bayes Gaussian để so sánh, dù không tối ưu cho văn bản rời rạc.",
    },
    "LogisticRegression + TfidfVectorizer": {
        "Mục đích": "Mô hình baseline không phải Bayes để đối chiếu hiệu quả.",
    },
}


# ============================================================
# HÀM XỬ LÝ DỮ LIỆU
# ============================================================

def normalize_label(value: Any) -> str:
    """
    Chuẩn hóa nhiều kiểu nhãn về 2 lớp thống nhất:
    - Dương tính
    - Âm tính
    """
    text = str(value).strip().lower()

    positive_values = {
        "tích cực", "tich cuc", "positive", "pos", "dương tính", "duong tinh", "1",
        "good", "hay", "tốt", "tot",
    }
    negative_values = {
        "tiêu cực", "tieu cuc", "negative", "neg", "âm tính", "am tinh", "0",
        "bad", "dở", "do", "tệ", "te",
    }

    if text in positive_values or "tích" in text or "dương" in text or "positive" in text:
        return POSITIVE_LABEL
    if text in negative_values or "tiêu" in text or "âm" in text or "negative" in text:
        return NEGATIVE_LABEL

    raise ValueError(
        f"Nhãn không hợp lệ: {value!r}. "
        "Hãy dùng Tích cực/Tiêu cực hoặc Dương tính/Âm tính."
    )


def tokenize_for_stats(text: str) -> List[str]:
    """
    Tách từ đơn giản để thống kê và giải thích.
    """
    return re.findall(r"[a-zA-ZÀ-ỹ0-9]+", str(text).lower())


def read_dataset(csv_path: Path) -> "pd.DataFrame":
    """
    Đọc dữ liệu CSV. File phải có hai cột:
    - text
    - label

    Lưu ý: Dữ liệu văn bản chỉ giả định tiếng Anh. Ngôn ngữ khác có thể cho kết quả không chính xác.
    """
    if IMPORT_ERROR:
        raise RuntimeError(
            "Máy chưa cài đủ thư viện.\n"
            "Hãy chạy: pip install -r requirements.txt\n\n"
            f"Lỗi import: {IMPORT_ERROR}"
        )

    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {csv_path}")

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="utf-8")

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("File CSV phải có đúng hai cột bắt buộc: text,label")

    df = df[["text", "label"]].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].apply(normalize_label)

    df = df[df["text"] != ""].reset_index(drop=True)

    if len(df) < 10:
        raise ValueError("Dữ liệu quá ít. Nên có tối thiểu 10 mẫu để chạy demo.")

    labels = set(df["label"].unique())
    if labels != {NEGATIVE_LABEL, POSITIVE_LABEL}:
        raise ValueError(
            f"Dữ liệu phải có đủ hai lớp {LABEL_ORDER}. "
            f"Hiện có: {sorted(labels)}"
        )

    return df


def dataset_stats(df: "pd.DataFrame") -> Dict[str, str]:
    lengths = df["text"].apply(lambda x: len(tokenize_for_stats(x)))
    counts = df["label"].value_counts()

    vocab = set()
    for text in df["text"]:
        vocab.update(tokenize_for_stats(text))

    return {
        "Số mẫu": str(len(df)),
        "Số lớp": "2",
        "Số mẫu Dương tính": str(int(counts.get(POSITIVE_LABEL, 0))),
        "Số mẫu Âm tính": str(int(counts.get(NEGATIVE_LABEL, 0))),
        "Kích thước từ vựng thô": str(len(vocab)),
        "Độ dài trung bình": f"{lengths.mean():.2f} từ/mẫu",
        "Độ dài trung vị": f"{lengths.median():.0f} từ/mẫu",
        "Độ dài ngắn nhất": f"{int(lengths.min())} từ",
        "Độ dài dài nhất": f"{int(lengths.max())} từ",
    }


class DenseTransformer(BaseEstimator, TransformerMixin):
    """
    GaussianNB yêu cầu ma trận dense.
    CountVectorizer/TfidfVectorizer trả về sparse matrix.
    Lớp này chuyển sparse matrix sang dense matrix để GaussianNB chạy được.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "toarray"):
            return X.toarray()
        return X


# ============================================================
# LỚP CHẠY THỰC NGHIỆM
# ============================================================

class BayesExperiment:
    def __init__(self):
        self.df: Optional["pd.DataFrame"] = None
        self.results_df: Optional["pd.DataFrame"] = None
        self.details: Dict[str, Dict[str, Any]] = {}
        self.best_estimators: Dict[str, Pipeline] = {}
        self.best_model_name: str = ""
        self.best_model: Optional[Pipeline] = None

    def load_data(self, csv_path: Path) -> "pd.DataFrame":
        self.df = read_dataset(csv_path)
        self.results_df = None
        self.details = {}
        self.best_estimators = {}
        self.best_model_name = ""
        self.best_model = None
        return self.df

    def build_model_grids(self) -> List[Tuple[str, Pipeline, Dict[str, List[Any]]]]:
        """
        Các mô hình và lưới tham số dùng trong thực nghiệm.
        """
        token_pattern = r"(?u)\b\w+\b"

        models = []

        models.append((
            "MultinomialNB + CountVectorizer",
            Pipeline([
                ("vectorizer", CountVectorizer(lowercase=True, token_pattern=token_pattern)),
                ("classifier", MultinomialNB()),
            ]),
            {
                "vectorizer__ngram_range": [(1, 1), (1, 2)],
                "vectorizer__min_df": [1, 2],
                "vectorizer__stop_words": [None, VIETNAMESE_STOPWORDS],
                "classifier__alpha": [0.1, 0.5, 1.0, 2.0],
            },
        ))

        models.append((
            "MultinomialNB + TfidfVectorizer",
            Pipeline([
                ("vectorizer", TfidfVectorizer(lowercase=True, token_pattern=token_pattern)),
                ("classifier", MultinomialNB()),
            ]),
            {
                "vectorizer__ngram_range": [(1, 1), (1, 2)],
                "vectorizer__min_df": [1, 2],
                "vectorizer__stop_words": [None, VIETNAMESE_STOPWORDS],
                "classifier__alpha": [0.1, 0.5, 1.0, 2.0],
            },
        ))

        models.append((
            "BernoulliNB + Binary CountVectorizer",
            Pipeline([
                ("vectorizer", CountVectorizer(lowercase=True, token_pattern=token_pattern, binary=True)),
                ("classifier", BernoulliNB()),
            ]),
            {
                "vectorizer__ngram_range": [(1, 1), (1, 2)],
                "vectorizer__min_df": [1, 2],
                "vectorizer__stop_words": [None, VIETNAMESE_STOPWORDS],
                "classifier__alpha": [0.1, 0.5, 1.0, 2.0],
            },
        ))

        models.append((
            "GaussianNB + TfidfVectorizer",
            Pipeline([
                ("vectorizer", TfidfVectorizer(lowercase=True, token_pattern=token_pattern)),
                ("to_dense", DenseTransformer()),
                ("classifier", GaussianNB()),
            ]),
            {
                "vectorizer__ngram_range": [(1, 1), (1, 2)],
                "vectorizer__min_df": [1, 2],
                "vectorizer__stop_words": [None],
                "classifier__var_smoothing": [1e-9, 1e-8, 1e-7],
            },
        ))

        models.append((
            "LogisticRegression + TfidfVectorizer",
            Pipeline([
                ("vectorizer", TfidfVectorizer(lowercase=True, token_pattern=token_pattern)),
                ("classifier", LogisticRegression(max_iter=2000, solver="liblinear", random_state=DEFAULT_RANDOM_STATE)),
            ]),
            {
                "vectorizer__ngram_range": [(1, 1), (1, 2)],
                "vectorizer__min_df": [1, 2],
                "vectorizer__stop_words": [None, VIETNAMESE_STOPWORDS],
                "classifier__C": [0.5, 1.0, 2.0],
            },
        ))

        return models

    def run(
        self,
        test_size: float = DEFAULT_TEST_SIZE,
        cv_folds: int = DEFAULT_CV_FOLDS,
        random_state: int = DEFAULT_RANDOM_STATE,
        log_callback=None,
    ) -> "pd.DataFrame":
        """
        Chạy toàn bộ thực nghiệm.
        """
        if self.df is None:
            raise RuntimeError("Chưa tải dữ liệu CSV.")

        df = self.df.copy()
        X = df["text"]
        y = df["label"]

        min_class_count = int(y.value_counts().min())
        cv_folds = min(cv_folds, min_class_count)
        if cv_folds < 2:
            raise ValueError("Mỗi lớp phải có ít nhất 2 mẫu để chạy cross validation.")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        skf = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state,
        )

        scoring = {
            "accuracy": "accuracy",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
            "f1_macro": "f1_macro",
            "roc_auc": "roc_auc",
        }

        rows = []
        self.details = {}
        self.best_estimators = {}

        model_grids = self.build_model_grids()

        for index, (model_name, pipeline, param_grid) in enumerate(model_grids, start=1):
            if log_callback:
                log_callback(f"[{index}/{len(model_grids)}] Đang GridSearchCV: {model_name}")

            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring="f1_macro",
                cv=skf,
                n_jobs=-1,
                refit=True,
            )
            grid.fit(X_train, y_train)

            best_estimator = grid.best_estimator_
            self.best_estimators[model_name] = best_estimator

            if log_callback:
                log_callback(f"    Best params: {grid.best_params_}")

            # Cross validation trên toàn bộ dữ liệu bằng mô hình tốt nhất.
            if log_callback:
                log_callback("    Đang tính 5-fold Stratified Cross Validation...")

            cv_scores = cross_validate(
                best_estimator,
                X,
                y,
                cv=skf,
                scoring=scoring,
                n_jobs=-1,
                error_score=np.nan,
            )

            # Đánh giá trên tập test.
            y_pred = best_estimator.predict(X_test)
            y_proba = self.safe_predict_proba(best_estimator, X_test)
            test_roc_auc = self.compute_roc_auc(y_test, y_proba, best_estimator)

            test_accuracy = accuracy_score(y_test, y_pred)
            test_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
            test_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
            test_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

            cm = confusion_matrix(y_test, y_pred, labels=LABEL_ORDER)
            report = classification_report(
                y_test,
                y_pred,
                labels=LABEL_ORDER,
                target_names=LABEL_ORDER,
                zero_division=0,
            )

            wrong_examples = []
            for text, true_label, pred_label in zip(X_test, y_test, y_pred):
                if true_label != pred_label:
                    wrong_examples.append({
                        "text": text,
                        "true_label": true_label,
                        "predicted_label": pred_label,
                    })

            vectorizer = best_estimator.named_steps.get("vectorizer")
            vocabulary_size = len(vectorizer.vocabulary_) if hasattr(vectorizer, "vocabulary_") else 0

            row = {
                "model": model_name,
                "best_params": json.dumps(grid.best_params_, ensure_ascii=False),
                "vocabulary_size": vocabulary_size,

                "cv_accuracy_mean": float(np.nanmean(cv_scores["test_accuracy"])),
                "cv_accuracy_std": float(np.nanstd(cv_scores["test_accuracy"])),
                "cv_precision_macro_mean": float(np.nanmean(cv_scores["test_precision_macro"])),
                "cv_recall_macro_mean": float(np.nanmean(cv_scores["test_recall_macro"])),
                "cv_f1_macro_mean": float(np.nanmean(cv_scores["test_f1_macro"])),
                "cv_roc_auc_mean": float(np.nanmean(cv_scores["test_roc_auc"])),
                "cv_roc_auc_std": float(np.nanstd(cv_scores["test_roc_auc"])),

                "test_accuracy": float(test_accuracy),
                "test_precision_macro": float(test_precision),
                "test_recall_macro": float(test_recall),
                "test_f1_macro": float(test_f1),
                "test_roc_auc": None if test_roc_auc is None else float(test_roc_auc),
                "wrong_count": len(wrong_examples),
            }
            rows.append(row)

            self.details[model_name] = {
                "best_estimator": best_estimator,
                "best_params": grid.best_params_,
                "confusion_matrix": cm,
                "classification_report": report,
                "wrong_examples": wrong_examples,
                "row": row,
                "reference": REPORT_REFERENCE_RESULTS.get(model_name, {}),
            }

            if log_callback:
                auc_text = "N/A" if test_roc_auc is None else f"{test_roc_auc:.4f}"
                log_callback(
                    f"    Test: Accuracy={test_accuracy:.4f}, "
                    f"Precision={test_precision:.4f}, Recall={test_recall:.4f}, "
                    f"F1={test_f1:.4f}, ROC-AUC={auc_text}, Sai={len(wrong_examples)}"
                )

        results = pd.DataFrame(rows)
        results = results.sort_values(
            by=["test_f1_macro", "test_accuracy", "cv_f1_macro_mean"],
            ascending=False,
        ).reset_index(drop=True)

        self.results_df = results
        self.best_model_name = str(results.iloc[0]["model"])
        self.best_model = self.best_estimators[self.best_model_name]

        if log_callback:
            log_callback(f"Hoàn thành. Mô hình tốt nhất: {self.best_model_name}")

        return results

    @staticmethod
    def safe_predict_proba(estimator, X):
        if hasattr(estimator, "predict_proba"):
            try:
                return estimator.predict_proba(X)
            except Exception:
                return None
        return None

    @staticmethod
    def compute_roc_auc(y_true, y_proba, estimator):
        if y_proba is None:
            return None

        try:
            classes = list(estimator.classes_)
            pos_idx = classes.index(POSITIVE_LABEL)
            y_score = y_proba[:, pos_idx]
            y_true_binary = np.array([1 if label == POSITIVE_LABEL else 0 for label in y_true])
            return roc_auc_score(y_true_binary, y_score)
        except Exception:
            return None

    def train_quick_default_model(self):
        """
        Huấn luyện nhanh mô hình chính nếu người dùng muốn dự đoán
        nhưng chưa bấm chạy thực nghiệm đầy đủ.
        """
        if self.df is None:
            raise RuntimeError("Chưa tải dữ liệu CSV.")

        model = Pipeline([
            ("vectorizer", CountVectorizer(
                lowercase=True,
                token_pattern=r"(?u)\b\w+\b",
                ngram_range=(1, 2),
                min_df=1,
            )),
            ("classifier", MultinomialNB(alpha=1.0)),
        ])

        model.fit(self.df["text"], self.df["label"])

        self.best_model_name = "MultinomialNB + CountVectorizer - huấn luyện nhanh"
        self.best_model = model
        self.best_estimators[self.best_model_name] = model

        return model

    def predict_text(self, text: str, model_name: Optional[str] = None):
        if self.df is None and self.best_model is None:
            raise RuntimeError("Chưa có dữ liệu hoặc mô hình. Hãy tải CSV trước.")

        if not text.strip():
            raise ValueError("Văn bản dự đoán không được để trống.")

        if model_name and model_name in self.best_estimators:
            model = self.best_estimators[model_name]
            used_name = model_name
        else:
            if self.best_model is None:
                model = self.train_quick_default_model()
            else:
                model = self.best_model
            used_name = self.best_model_name

        pred = model.predict([text])[0]

        proba_pairs = []
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba([text])[0]
                proba_pairs = list(zip(model.classes_, proba))
                proba_pairs.sort(key=lambda x: x[1], reverse=True)
            except Exception:
                pass

        explanation = self.explain_prediction(model, text)

        return {
            "model_name": used_name,
            "prediction": pred,
            "probabilities": proba_pairs,
            "tokens": tokenize_for_stats(text),
            "explanation": explanation,
        }

    def explain_prediction(self, model: Pipeline, text: str, top_k: int = 15) -> List[Tuple[str, float, str]]:
        """
        Giải thích đơn giản các từ/cụm từ ảnh hưởng tới dự đoán.

        Với Naive Bayes:
            dùng chênh lệch log P(feature|Dương tính) - log P(feature|Âm tính).
        Với Logistic Regression:
            dùng hệ số trọng số của mô hình.
        """
        try:
            vectorizer = model.named_steps.get("vectorizer")
            classifier = model.named_steps.get("classifier")

            if vectorizer is None or classifier is None:
                return []

            X_vec = vectorizer.transform([text])
            feature_names = np.array(vectorizer.get_feature_names_out())

            if hasattr(X_vec, "toarray"):
                arr = X_vec.toarray()[0]
            else:
                arr = X_vec[0]

            active_indices = np.where(arr > 0)[0]
            items = []

            if hasattr(classifier, "feature_log_prob_"):
                classes = list(classifier.classes_)
                if POSITIVE_LABEL in classes and NEGATIVE_LABEL in classes:
                    pos_idx = classes.index(POSITIVE_LABEL)
                    neg_idx = classes.index(NEGATIVE_LABEL)
                    diff = classifier.feature_log_prob_[pos_idx] - classifier.feature_log_prob_[neg_idx]

                    for idx in active_indices:
                        word = feature_names[idx]
                        score = float(diff[idx] * arr[idx])
                        tendency = POSITIVE_LABEL if score > 0 else NEGATIVE_LABEL if score < 0 else "Trung tính"
                        items.append((word, score, tendency))

            elif hasattr(classifier, "coef_"):
                # Logistic Regression nhị phân
                coef = classifier.coef_[0]
                classes = list(classifier.classes_)
                # Trong binary sklearn, coef_ thường ứng với classes_[1]
                positive_class = classes[1] if len(classes) > 1 else POSITIVE_LABEL
                sign = 1.0 if positive_class == POSITIVE_LABEL else -1.0

                for idx in active_indices:
                    word = feature_names[idx]
                    score = float(coef[idx] * arr[idx] * sign)
                    tendency = POSITIVE_LABEL if score > 0 else NEGATIVE_LABEL if score < 0 else "Trung tính"
                    items.append((word, score, tendency))

            items.sort(key=lambda x: abs(x[1]), reverse=True)
            return items[:top_k]

        except Exception:
            return []

    def save_model(self, path: Path):
        if joblib is None:
            raise RuntimeError("Thiếu joblib. Hãy chạy: pip install joblib")
        if self.best_model is None:
            raise RuntimeError("Chưa có mô hình để lưu. Hãy chạy thực nghiệm hoặc huấn luyện nhanh trước.")

        payload = {
            "model_name": self.best_model_name,
            "model": self.best_model,
            "labels": LABEL_ORDER,
        }
        joblib.dump(payload, path)

    def load_model(self, path: Path):
        if joblib is None:
            raise RuntimeError("Thiếu joblib. Hãy chạy: pip install joblib")
        payload = joblib.load(path)

        if isinstance(payload, dict) and "model" in payload:
            self.best_model = payload["model"]
            self.best_model_name = payload.get("model_name", "Mô hình đã tải")
        else:
            self.best_model = payload
            self.best_model_name = "Mô hình đã tải"

        self.best_estimators[self.best_model_name] = self.best_model


# ============================================================
# GIAO DIỆN
# ============================================================

class FullExperimentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Naive Bayes - Thực nghiệm, đánh giá và dự đoán văn bản")
        self.root.geometry("1280x780")
        self.root.minsize(1120, 680)

        self.experiment = BayesExperiment()
        self.csv_path: Optional[Path] = None

        self.csv_path_var = tk.StringVar(value=str(Path(__file__).resolve().parent / DEFAULT_CSV))
        self.test_size_var = tk.StringVar(value=str(DEFAULT_TEST_SIZE))
        self.cv_folds_var = tk.StringVar(value=str(DEFAULT_CV_FOLDS))
        self.random_state_var = tk.StringVar(value=str(DEFAULT_RANDOM_STATE))
        self.model_choice_var = tk.StringVar(value="")

        self.build_ui()
        self.load_default_if_exists()

    # ------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------

    def build_ui(self):
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Times New Roman", 17, "bold"))
        style.configure("Heading.TLabel", font=("Times New Roman", 12, "bold"))
        style.configure("Normal.TLabel", font=("Times New Roman", 12))
        style.configure("TButton", font=("Times New Roman", 11))

        title = ttk.Label(
            self.root,
            text="DEMO PHÂN LOẠI CẢM XÚC VĂN BẢN BẰNG BAYES",
            style="Title.TLabel",
        )
        title.pack(pady=(8, 2))

        subtitle = ttk.Label(
            self.root,
            text="Chạy thực nghiệm, đánh giá kết quả, lưu mô hình và dự đoán văn bản mới",
            style="Normal.TLabel",
        )
        subtitle.pack(pady=(0, 8))

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        self.tab_data = ttk.Frame(self.notebook, padding=8)
        self.tab_experiment = ttk.Frame(self.notebook, padding=8)
        self.tab_prediction = ttk.Frame(self.notebook, padding=8)
        self.tab_detail = ttk.Frame(self.notebook, padding=8)

        self.notebook.add(self.tab_data, text="Dữ liệu")
        self.notebook.add(self.tab_experiment, text="Thực nghiệm & đánh giá")
        self.notebook.add(self.tab_prediction, text="Dự đoán văn bản")
        self.notebook.add(self.tab_detail, text="Chi tiết / Xuất file")

        self.build_data_tab()
        self.build_experiment_tab()
        self.build_prediction_tab()
        self.build_detail_tab()

    def build_data_tab(self):
        top = ttk.LabelFrame(self.tab_data, text="Tải dữ liệu CSV", padding=10)
        top.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(top, text="Đường dẫn CSV:", style="Heading.TLabel").pack(anchor="w")

        path_row = ttk.Frame(top)
        path_row.pack(fill=tk.X, pady=4)

        ttk.Entry(path_row, textvariable=self.csv_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(path_row, text="Chọn CSV", command=self.choose_csv).pack(side=tk.LEFT, padx=4)
        ttk.Button(path_row, text="Tải lại dữ liệu", command=self.reload_csv).pack(side=tk.LEFT, padx=4)

        middle = ttk.PanedWindow(self.tab_data, orient=tk.HORIZONTAL)
        middle.pack(fill=tk.BOTH, expand=True)

        stats_frame = ttk.LabelFrame(middle, text="Thống kê dữ liệu", padding=8)
        preview_frame = ttk.LabelFrame(middle, text="Xem trước dữ liệu", padding=8)

        middle.add(stats_frame, weight=1)
        middle.add(preview_frame, weight=3)

        self.stats_text = tk.Text(stats_frame, wrap=tk.WORD, font=("Consolas", 11), height=18)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        self.stats_text.config(state=tk.DISABLED)

        columns = ("text", "label")
        self.data_tree = ttk.Treeview(preview_frame, columns=columns, show="headings", height=18)
        self.data_tree.heading("text", text="text")
        self.data_tree.heading("label", text="label")
        self.data_tree.column("text", width=800, minwidth=800, stretch=False)
        self.data_tree.column("label", width=140, minwidth=140, anchor="center", stretch=False)

        data_x = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, command=self.data_tree.xview)
        data_y = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        self.data_tree.configure(xscrollcommand=data_x.set, yscrollcommand=data_y.set)

        self.data_tree.grid(row=0, column=0, sticky="nsew")
        data_y.grid(row=0, column=1, sticky="ns")
        data_x.grid(row=1, column=0, sticky="ew")

        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)

    def build_experiment_tab(self):
        control = ttk.LabelFrame(self.tab_experiment, text="Thiết lập và chạy thực nghiệm", padding=8)
        control.pack(fill=tk.X, pady=(0, 8))

        row1 = ttk.Frame(control)
        row1.pack(fill=tk.X)

        self.add_labeled_entry(row1, "Test size:", self.test_size_var, width=8)
        self.add_labeled_entry(row1, "CV folds:", self.cv_folds_var, width=8)
        self.add_labeled_entry(row1, "Random state:", self.random_state_var, width=8)

        ttk.Button(row1, text="Chạy thực nghiệm đầy đủ", command=self.run_experiment_threaded).pack(side=tk.LEFT, padx=8)
        ttk.Button(row1, text="Xuất kết quả CSV", command=self.export_results).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="Lưu mô hình tốt nhất", command=self.save_model).pack(side=tk.LEFT, padx=4)

        # Bảng kết quả có thanh kéo ngang/dọc, cột giữ rộng.
        result_frame = ttk.LabelFrame(self.tab_experiment, text="Bảng kết quả so sánh mô hình", padding=8)
        result_frame.pack(fill=tk.BOTH, expand=True)

        columns = (
            "model",
            "vocab",
            "cv_acc",
            "cv_acc_std",
            "cv_precision",
            "cv_recall",
            "cv_f1",
            "cv_auc",
            "test_acc",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_auc",
            "wrong",
            "best_params",
        )

        headings = {
            "model": "Mô hình",
            "vocab": "Vocab",
            "cv_acc": "CV Accuracy",
            "cv_acc_std": "CV Acc Std",
            "cv_precision": "CV Precision",
            "cv_recall": "CV Recall",
            "cv_f1": "CV F1",
            "cv_auc": "CV ROC-AUC",
            "test_acc": "Test Accuracy",
            "test_precision": "Test Precision",
            "test_recall": "Test Recall",
            "test_f1": "Test F1",
            "test_auc": "Test ROC-AUC",
            "wrong": "Số mẫu sai",
            "best_params": "Best params",
        }

        widths = {
            "model": 360,
            "vocab": 80,
            "cv_acc": 120,
            "cv_acc_std": 110,
            "cv_precision": 120,
            "cv_recall": 110,
            "cv_f1": 100,
            "cv_auc": 120,
            "test_acc": 120,
            "test_precision": 125,
            "test_recall": 115,
            "test_f1": 100,
            "test_auc": 120,
            "wrong": 100,
            "best_params": 760,
        }

        tree_box = ttk.Frame(result_frame)
        tree_box.pack(fill=tk.BOTH, expand=True)

        self.result_tree = ttk.Treeview(tree_box, columns=columns, show="headings", height=16)
        result_x = ttk.Scrollbar(tree_box, orient=tk.HORIZONTAL, command=self.result_tree.xview)
        result_y = ttk.Scrollbar(tree_box, orient=tk.VERTICAL, command=self.result_tree.yview)
        self.result_tree.configure(xscrollcommand=result_x.set, yscrollcommand=result_y.set)

        for col in columns:
            self.result_tree.heading(col, text=headings[col])
            self.result_tree.column(
                col,
                width=widths[col],
                minwidth=widths[col],
                anchor="center" if col not in {"model", "best_params"} else "w",
                stretch=False,
            )

        self.result_tree.grid(row=0, column=0, sticky="nsew")
        result_y.grid(row=0, column=1, sticky="ns")
        result_x.grid(row=1, column=0, sticky="ew")
        tree_box.rowconfigure(0, weight=1)
        tree_box.columnconfigure(0, weight=1)

        self.result_tree.bind("<<TreeviewSelect>>", self.show_selected_detail)

        log_frame = ttk.LabelFrame(self.tab_experiment, text="Tiến trình chạy", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self.log_text = tk.Text(log_frame, wrap=tk.WORD, font=("Consolas", 10), height=9)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)

    def build_prediction_tab(self):
        top = ttk.LabelFrame(self.tab_prediction, text="Predict a new review", padding=10)
        top.pack(fill=tk.BOTH, expand=True)

        model_row = ttk.Frame(top)
        model_row.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(model_row, text="Model:", style="Heading.TLabel").pack(side=tk.LEFT)
        self.model_combo = ttk.Combobox(
            model_row,
            textvariable=self.model_choice_var,
            state="readonly",
            width=60,
        )
        self.model_combo.pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True)

        ttk.Button(model_row, text="Load saved model", command=self.load_model).pack(side=tk.LEFT, padx=4)

        ttk.Label(top, text="Enter review to classify:", style="Heading.TLabel").pack(anchor="w")

        self.input_text = tk.Text(top, wrap=tk.WORD, font=("Times New Roman", 13), height=7)
        self.input_text.pack(fill=tk.X, pady=6)
        self.input_text.insert(tk.END, "The movie had an engaging story, strong performances, and a moving ending.")

        btn_row = ttk.Frame(top)
        btn_row.pack(fill=tk.X, pady=4)

        ttk.Button(btn_row, text="Predict", command=self.predict_text).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Positive example", command=self.example_positive).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Negative example", command=self.example_negative).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Clear", command=self.clear_prediction_input).pack(side=tk.LEFT, padx=4)

        self.prediction_label = ttk.Label(
            top,
            text="Result: Not predicted",
            font=("Times New Roman", 15, "bold"),
        )
        self.prediction_label.pack(anchor="w", pady=8)

        result_pane = ttk.PanedWindow(top, orient=tk.HORIZONTAL)
        result_pane.pack(fill=tk.BOTH, expand=True)

        pred_frame = ttk.LabelFrame(result_pane, text="Details", padding=8)
        explain_frame = ttk.LabelFrame(result_pane, text="Influential tokens", padding=8)

        result_pane.add(pred_frame, weight=1)
        result_pane.add(explain_frame, weight=1)

        self.prediction_text = tk.Text(pred_frame, wrap=tk.WORD, font=("Consolas", 10))
        self.prediction_text.pack(fill=tk.BOTH, expand=True)
        self.prediction_text.config(state=tk.DISABLED)

        self.explain_text = tk.Text(explain_frame, wrap=tk.WORD, font=("Consolas", 10))
        self.explain_text.pack(fill=tk.BOTH, expand=True)
        self.explain_text.config(state=tk.DISABLED)

    def build_detail_tab(self):
        top = ttk.Frame(self.tab_detail)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Chi tiết mô hình đang chọn:", style="Heading.TLabel").pack(side=tk.LEFT)
        self.detail_title = ttk.Label(top, text="Chưa chọn", style="Normal.TLabel")
        self.detail_title.pack(side=tk.LEFT, padx=8)

        ttk.Button(top, text="Xuất kết quả CSV", command=self.export_results).pack(side=tk.RIGHT, padx=4)
        ttk.Button(top, text="Lưu mô hình tốt nhất", command=self.save_model).pack(side=tk.RIGHT, padx=4)

        detail_frame = ttk.LabelFrame(self.tab_detail, text="Confusion Matrix, Classification Report và mẫu sai", padding=8)
        detail_frame.pack(fill=tk.BOTH, expand=True, pady=8)

        self.detail_text = tk.Text(detail_frame, wrap=tk.WORD, font=("Consolas", 10))
        detail_y = ttk.Scrollbar(detail_frame, orient=tk.VERTICAL, command=self.detail_text.yview)
        self.detail_text.configure(yscrollcommand=detail_y.set)

        self.detail_text.grid(row=0, column=0, sticky="nsew")
        detail_y.grid(row=0, column=1, sticky="ns")
        detail_frame.rowconfigure(0, weight=1)
        detail_frame.columnconfigure(0, weight=1)

        self.detail_text.config(state=tk.DISABLED)

    def add_labeled_entry(self, parent, label, variable, width=10):
        frame = ttk.Frame(parent)
        frame.pack(side=tk.LEFT, padx=4)
        ttk.Label(frame, text=label, style="Normal.TLabel").pack(side=tk.LEFT)
        ttk.Entry(frame, textvariable=variable, width=width).pack(side=tk.LEFT, padx=2)

    # ------------------------------------------------------------
    # Data actions
    # ------------------------------------------------------------

    def load_default_if_exists(self):
        path = Path(self.csv_path_var.get())
        if path.exists():
            self.reload_csv(show_message=False)
        else:
            self.set_text(self.stats_text, "Chưa tìm thấy CSV mặc định. Hãy bấm 'Chọn CSV' để tải dữ liệu.")

    def choose_csv(self):
        path = filedialog.askopenfilename(
            title="Chọn file dữ liệu CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.csv_path_var.set(path)
            self.reload_csv()

    def reload_csv(self, show_message=True):
        try:
            path = Path(self.csv_path_var.get())
            self.csv_path = path
            df = self.experiment.load_data(path)

            stats = dataset_stats(df)
            text = "\n".join(f"{k:28s}: {v}" for k, v in stats.items())
            self.set_text(self.stats_text, text)

            self.fill_data_preview(df)
            self.clear_result_table()
            self.set_text(self.detail_text, "")
            self.set_text(self.log_text, f"Đã tải dữ liệu: {path}\n{text}\n")
            self.update_model_combo()

            if show_message:
                messagebox.showinfo("Thành công", "Đã tải dữ liệu CSV.")

        except Exception as e:
            self.set_text(self.stats_text, traceback.format_exc())
            messagebox.showerror("Lỗi tải dữ liệu", str(e))

    def fill_data_preview(self, df: "pd.DataFrame"):
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)

        for _, row in df.head(300).iterrows():
            self.data_tree.insert("", tk.END, values=(row["text"], row["label"]))

    # ------------------------------------------------------------
    # Experiment actions
    # ------------------------------------------------------------

    def run_experiment_threaded(self):
        thread = threading.Thread(target=self.run_experiment, daemon=True)
        thread.start()

    def run_experiment(self):
        try:
            test_size = float(self.test_size_var.get())
            cv_folds = int(self.cv_folds_var.get())
            random_state = int(self.random_state_var.get())

            if not (0.05 <= test_size <= 0.5):
                raise ValueError("Test size nên nằm trong khoảng 0.05 đến 0.5.")

            self.root.after(0, lambda: self.root.config(cursor="watch"))
            self.root.after(0, lambda: self.set_text(self.log_text, "Bắt đầu chạy thực nghiệm...\n"))
            self.root.after(0, self.clear_result_table)

            results = self.experiment.run(
                test_size=test_size,
                cv_folds=cv_folds,
                random_state=random_state,
                log_callback=self.append_log_threadsafe,
            )

            self.root.after(0, lambda: self.fill_result_table(results))
            self.root.after(0, self.update_model_combo)
            self.root.after(0, lambda: messagebox.showinfo("Hoàn thành", "Đã chạy xong thực nghiệm."))

        except Exception as e:
            self.root.after(0, lambda: self.set_text(self.log_text, traceback.format_exc()))
            self.root.after(0, lambda: messagebox.showerror("Lỗi chạy thực nghiệm", str(e)))
        finally:
            self.root.after(0, lambda: self.root.config(cursor=""))

    def clear_result_table(self):
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)

    def fill_result_table(self, results: "pd.DataFrame"):
        self.clear_result_table()

        for _, row in results.iterrows():
            auc_text = "N/A" if row["test_roc_auc"] is None or pd.isna(row["test_roc_auc"]) else f"{row['test_roc_auc']:.4f}"

            values = (
                row["model"],
                int(row["vocabulary_size"]),
                f"{row['cv_accuracy_mean']:.4f}",
                f"{row['cv_accuracy_std']:.4f}",
                f"{row['cv_precision_macro_mean']:.4f}",
                f"{row['cv_recall_macro_mean']:.4f}",
                f"{row['cv_f1_macro_mean']:.4f}",
                f"{row['cv_roc_auc_mean']:.4f}",
                f"{row['test_accuracy']:.4f}",
                f"{row['test_precision_macro']:.4f}",
                f"{row['test_recall_macro']:.4f}",
                f"{row['test_f1_macro']:.4f}",
                auc_text,
                int(row["wrong_count"]),
                row["best_params"],
            )
            self.result_tree.insert("", tk.END, values=values)

        children = self.result_tree.get_children()
        if children:
            self.result_tree.selection_set(children[0])
            self.show_selected_detail()

    def show_selected_detail(self, event=None):
        selected = self.result_tree.selection()
        if not selected:
            return

        values = self.result_tree.item(selected[0], "values")
        model_name = values[0]
        self.show_model_detail(model_name)
        self.notebook.select(self.tab_detail)

    def show_model_detail(self, model_name: str):
        detail = self.experiment.details.get(model_name)
        if not detail:
            return

        self.detail_title.config(text=model_name)

        row = detail["row"]
        cm = detail["confusion_matrix"]
        wrong = detail["wrong_examples"]
        ref = detail.get("reference", {})

        lines = []
        lines.append(f"MÔ HÌNH: {model_name}")
        lines.append("=" * 100)
        lines.append("")

        if ref:
            lines.append("Mục đích dùng mô hình:")
            for k, v in ref.items():
                lines.append(f"- {k}: {v}")
            lines.append("")

        lines.append("1. Tham số tốt nhất do GridSearchCV tìm được")
        lines.append(json.dumps(detail["best_params"], ensure_ascii=False, indent=2))
        lines.append("")

        lines.append("2. Bảng chỉ số")
        lines.append(f"Vocabulary size        : {row['vocabulary_size']}")
        lines.append(f"CV Accuracy mean/std   : {row['cv_accuracy_mean']:.4f} / {row['cv_accuracy_std']:.4f}")
        lines.append(f"CV Precision macro     : {row['cv_precision_macro_mean']:.4f}")
        lines.append(f"CV Recall macro        : {row['cv_recall_macro_mean']:.4f}")
        lines.append(f"CV F1 macro            : {row['cv_f1_macro_mean']:.4f}")
        lines.append(f"CV ROC-AUC mean/std    : {row['cv_roc_auc_mean']:.4f} / {row['cv_roc_auc_std']:.4f}")
        lines.append(f"Test Accuracy          : {row['test_accuracy']:.4f}")
        lines.append(f"Test Precision macro   : {row['test_precision_macro']:.4f}")
        lines.append(f"Test Recall macro      : {row['test_recall_macro']:.4f}")
        lines.append(f"Test F1 macro          : {row['test_f1_macro']:.4f}")
        lines.append(f"Test ROC-AUC           : {row['test_roc_auc']}")
        lines.append(f"Số mẫu dự đoán sai     : {row['wrong_count']}")
        lines.append("")

        lines.append("3. Confusion Matrix")
        lines.append(f"Thứ tự nhãn: {LABEL_ORDER}")
        lines.append("")
        lines.append("                      Dự đoán Âm tính     Dự đoán Dương tính")
        lines.append(f"Thực tế Âm tính        {cm[0][0]:8d}              {cm[0][1]:8d}")
        lines.append(f"Thực tế Dương tính     {cm[1][0]:8d}              {cm[1][1]:8d}")
        lines.append("")

        lines.append("4. Classification Report")
        lines.append(detail["classification_report"])
        lines.append("")

        lines.append("5. Các mẫu dự đoán sai trên tập test")
        lines.append("-" * 100)
        if not wrong:
            lines.append("Không có mẫu dự đoán sai trong lần chạy này.")
        else:
            for i, item in enumerate(wrong, start=1):
                lines.append(f"[{i}] True: {item['true_label']} | Predicted: {item['predicted_label']}")
                lines.append(f"    Text: {item['text']}")
                lines.append("")

        self.set_text(self.detail_text, "\n".join(lines))

    # ------------------------------------------------------------
    # Prediction actions
    # ------------------------------------------------------------

    def update_model_combo(self):
        names = list(self.experiment.best_estimators.keys())
        self.model_combo["values"] = names

        if self.experiment.best_model_name:
            self.model_choice_var.set(self.experiment.best_model_name)
        elif names:
            self.model_choice_var.set(names[0])
        else:
            self.model_choice_var.set("")

    def display_label(self, label: str) -> str:
        mapping = {
            POSITIVE_LABEL: "Positive",
            NEGATIVE_LABEL: "Negative",
            "Trung tính": "Neutral",
            "Positive": "Positive",
            "Negative": "Negative",
            "Neutral": "Neutral",
        }
        return mapping.get(label, label)

    def predict_text(self):
        try:
            text = self.input_text.get("1.0", tk.END).strip()
            model_name = self.model_choice_var.get().strip() or None

            result = self.experiment.predict_text(text, model_name)

            prediction_label = self.display_label(result["prediction"])

            self.prediction_label.config(text=f"Result: {prediction_label}")

            lines = []
            lines.append(f"Model: {result['model_name']}")
            lines.append(f"Prediction: {prediction_label}")
            lines.append("")
            lines.append("Probabilities:")
            if result["probabilities"]:
                for label, prob in result["probabilities"]:
                    lines.append(f"- {self.display_label(label)}: {prob * 100:.2f}%")
            else:
                lines.append("Model does not support predict_proba.")
            lines.append("")
            lines.append("Tokens after basic preprocessing:")
            lines.append(", ".join(result["tokens"]))

            explain_lines = []
            explain_lines.append("feature/token                          impact score       leans toward")
            explain_lines.append("-" * 78)
            if result["explanation"]:
                for feature, score, tendency in result["explanation"]:
                    explain_lines.append(
                        f"{feature:38s} {score:>14.6f}      {self.display_label(tendency)}"
                    )
            else:
                explain_lines.append("No explanation available for this model.")

            self.set_text(self.prediction_text, "\n".join(lines))
            self.set_text(self.explain_text, "\n".join(explain_lines))

        except Exception as e:
            self.set_text(self.prediction_text, traceback.format_exc())
            messagebox.showerror("Lỗi dự đoán", str(e))

    def example_positive(self):
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert(tk.END, "A great movie with natural acting and a touching story.")

    def example_negative(self):
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert(tk.END, "A boring film with a weak script and terrible acting.")

    def clear_prediction_input(self):
        self.input_text.delete("1.0", tk.END)
        self.prediction_label.config(text="Result: Not predicted")
        self.set_text(self.prediction_text, "")
        self.set_text(self.explain_text, "")

    # ------------------------------------------------------------
    # Export / model actions
    # ------------------------------------------------------------

    def export_results(self):
        if self.experiment.results_df is None:
            messagebox.showwarning("Chưa có kết quả", "Hãy chạy thực nghiệm trước.")
            return

        path = filedialog.asksaveasfilename(
            title="Xuất kết quả thực nghiệm",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="ket_qua_thuc_nghiem_bayes.csv",
        )

        if not path:
            return

        self.experiment.results_df.to_csv(path, index=False, encoding="utf-8-sig")
        messagebox.showinfo("Đã xuất", f"Đã xuất kết quả vào:\n{path}")

    def save_model(self):
        try:
            if self.experiment.best_model is None:
                messagebox.showwarning(
                    "Chưa có mô hình",
                    "Hãy chạy thực nghiệm hoặc dự đoán thử để huấn luyện mô hình trước.",
                )
                return

            path = filedialog.asksaveasfilename(
                title="Lưu mô hình",
                defaultextension=".joblib",
                filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")],
                initialfile="bayes_best_model.joblib",
            )

            if not path:
                return

            self.experiment.save_model(Path(path))
            messagebox.showinfo("Đã lưu", f"Đã lưu mô hình vào:\n{path}")

        except Exception as e:
            messagebox.showerror("Lỗi lưu mô hình", str(e))

    def load_model(self):
        try:
            path = filedialog.askopenfilename(
                title="Tải mô hình đã lưu",
                filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")],
            )

            if not path:
                return

            self.experiment.load_model(Path(path))
            self.update_model_combo()
            messagebox.showinfo("Đã tải", f"Đã tải mô hình:\n{path}")

        except Exception as e:
            messagebox.showerror("Lỗi tải mô hình", str(e))

    # ------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------

    def set_text(self, widget: tk.Text, content: str):
        widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, content)
        widget.config(state=tk.DISABLED)

    def append_log(self, message: str):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def append_log_threadsafe(self, message: str):
        self.root.after(0, lambda: self.append_log(message))


def main():
    root = tk.Tk()
    app = FullExperimentApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
