# -*- coding: utf-8 -*-
"""
Chương trình Phân loại Cảm xúc Văn bản sử dụng thuật toán Naive Bayes.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import csv
import math
import re
import random
import pickle
from pathlib import Path

# ============================================================
# CẤU HÌNH CƠ BẢN
# ============================================================
DEFAULT_CSV = "du_lieu_nhan_xet_phim_bayes.csv"
POSITIVE_LABEL = "Dương tính"
NEGATIVE_LABEL = "Âm tính"

# Preprocessing and fallback configuration (Package 3)
ENGLISH_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "because", "as",
    "of", "to", "in", "on", "at", "for", "with", "by", "from", "about", "into",
    "over", "under", "again", "once", "than", "only", "own", "same", "such",
    "this", "that", "these", "those", "here", "there",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "doing", "have", "has", "had", "having",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "them",
    "my", "your", "his", "their", "our", "yours", "ours", "theirs", "its",
    "myself", "yourself", "himself", "herself", "itself", "ourselves", "themselves",
    "can", "could", "should", "would", "may", "might", "will", "just", "very",
    "too", "also", "s", "t", "not", "no", "never",
    "film", "movie"
}
NEGATION_WORDS = {"not", "no", "never", "n't"}

POS_LEXICON = {
    "good", "great", "excellent", "amazing", "wonderful", "love", "loved", "like",
    "liked", "fantastic", "best", "enjoy", "enjoyed", "engaging", "brilliant",
    "superb", "awesome", "positive", "satisfying", "beautiful", "memorable",
    "strong", "impressive", "perfect", "favorite", "fun", "charming",
    "heartwarming", "masterpiece", "delightful", "moving"
}
NEG_LEXICON = {
    "bad", "terrible", "awful", "boring", "worst", "hate", "hated", "poor", "weak",
    "disappointing", "disappointed", "messy", "dull", "annoying", "horrible",
    "mediocre", "negative", "unwatchable", "waste", "slow", "painful", "confusing",
    "forgettable", "predictable", "flawed", "stupid", "ridiculous", "tiring",
    "ugly", "cringe", "nonsense", "disaster", "lame"
}

SHORT_SENTENCE_MAX_TOKENS = 4
SHORT_SENTENCE_LEXICON_BOOST = 1.5
UNCERTAIN_PROB_THRESHOLD = 0.6
UNCERTAIN_MARGIN_THRESHOLD = 0.15
OOV_RATIO_THRESHOLD = 0.6
MIN_VALID_TOKENS = 2


# ============================================================
# CÁC HÀM TIỀN XỬ LÝ
# ============================================================

def normalize_label(value):
    """
    Chuẩn hóa nhãn về 2 loại duy nhất: Dương tính hoặc Âm tính.
    Nhận diện các từ khóa tương đương để đồng nhất dữ liệu thực tế.
    """
    text = str(value).strip().lower()
    
    # Tập các từ khóa biểu hiện sự tích cực
    positive_values = {"tích cực", "positive", "pos", "dương tính", "1", "good", "hay", "tốt"}
    # Tập các từ khóa biểu hiện sự tiêu cực
    negative_values = {"tiêu cực", "negative", "neg", "âm tính", "0", "bad", "dở", "tệ"}

    if text in positive_values or "tích" in text or "dương" in text or "positive" in text:
        return POSITIVE_LABEL
    if text in negative_values or "tiêu" in text or "âm" in text or "negative" in text:
        return NEGATIVE_LABEL
    
    raise ValueError(f"Nhãn không hợp lệ: '{value}'. Hãy dùng Tích cực/Tiêu cực hoặc Dương tính/Âm tính.")

def build_negation_bigrams(tokens):
    """Gom cụm phủ định: not good -> not_good."""
    result = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in NEGATION_WORDS and i + 1 < len(tokens):
            next_tok = tokens[i + 1]
            result.append(f"not_{next_tok}")
            i += 2
            continue
        result.append(token)
        i += 1
    return result


def tokenize(text):
    """
    Tách câu thành danh sách các từ (token) cho văn bản tiếng Anh.
    - Loại stopword.
    - Gom cụm phủ định đơn giản (not good -> not_good).
    """
    raw_tokens = re.findall(r"[a-zA-Z0-9]+", str(text).lower())
    if not raw_tokens:
        return []
    tokens = build_negation_bigrams(raw_tokens)
    tokens = [t for t in tokens if t not in ENGLISH_STOPWORDS]
    return tokens


def lexicon_score(tokens):
    """Tính điểm cảm xúc từ lexicon thủ công."""
    pos_hits = []
    neg_hits = []
    for token in tokens:
        if token.startswith("not_"):
            base = token[4:]
            if base in POS_LEXICON:
                neg_hits.append(token)
            elif base in NEG_LEXICON:
                pos_hits.append(token)
            continue
        if token in POS_LEXICON:
            pos_hits.append(token)
        elif token in NEG_LEXICON:
            neg_hits.append(token)
    score = len(pos_hits) - len(neg_hits)
    return score, pos_hits, neg_hits


def softmax_from_log_scores(log_scores):
    if not log_scores:
        return {}
    max_log = max(log_scores.values())
    exps = {c: math.exp(score - max_log) for c, score in log_scores.items()}
    sum_exps = sum(exps.values())
    return {c: (exps[c] / sum_exps) for c in exps}


# ============================================================
# MÔ HÌNH NAIVE BAYES
# ============================================================


# Chỉ giữ lại Multinomial Naive Bayes
class ManualNaiveBayes:
    """
    Cài đặt thuật toán Naive Bayes Multinomial.
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.vocab = set()
        self.vocab_size = 0
        self.class_log_prior = {}
        self.word_log_prob = {}              # log(P(w|c))
        self.classes = []
        self.default_log_prob = {}           # Xác suất cho từ lạ
        self.is_trained = False

    def fit(self, X, y):
        class_counts = {}
        word_counts = {}
        total_docs = len(X)

        for text, label in zip(X, y):
            if label not in class_counts:
                class_counts[label] = 0
                word_counts[label] = {}
            class_counts[label] += 1
            tokens = tokenize(text)
            for token in tokens:
                self.vocab.add(token)
                word_counts[label][token] = word_counts[label].get(token, 0) + 1

        self.classes = list(class_counts.keys())
        self.vocab_size = len(self.vocab)

        for c in self.classes:
            self.class_log_prior[c] = math.log(class_counts[c] / total_docs)
            self.word_log_prob[c] = {}
            total_words_in_c = sum(word_counts[c].values())
            denominator = total_words_in_c + (self.alpha * self.vocab_size)
            for word, count in word_counts[c].items():
                self.word_log_prob[c][word] = math.log((count + self.alpha) / denominator)
            self.default_log_prob[c] = math.log(self.alpha / denominator)
        self.is_trained = True

    def predict(self, text):
        if not self.is_trained:
            raise Exception("Mô hình chưa được huấn luyện!")
        tokens = tokenize(text)
        scores = {}
        for c in self.classes:
            score = self.class_log_prior[c]
            for token in tokens:
                score += self.word_log_prob[c].get(token, self.default_log_prob[c])
            scores[c] = score
        best_class = max(scores, key=scores.get)
        return best_class, scores

    def predict_proba(self, text):
        _, log_scores = self.predict(text)
        return softmax_from_log_scores(log_scores)

    def explain(self, text, top_k=15):
        if not self.is_trained or len(self.classes) < 2:
            return []
        tokens = tokenize(text)
        tokens = [t for t in tokens if t in self.vocab]
        pos_class, neg_class = self.classes[0], self.classes[1]
        if POSITIVE_LABEL in self.classes and NEGATIVE_LABEL in self.classes:
            pos_class, neg_class = POSITIVE_LABEL, NEGATIVE_LABEL
        token_counts = {}
        for t in tokens:
            token_counts[t] = token_counts.get(t, 0) + 1
        explanations = []
        for token, count in token_counts.items():
            prob_pos = self.word_log_prob[pos_class].get(token, self.default_log_prob[pos_class])
            prob_neg = self.word_log_prob[neg_class].get(token, self.default_log_prob[neg_class])
            diff = prob_pos - prob_neg
            impact_score = diff * count
            tendency = pos_class if impact_score > 0 else neg_class if impact_score < 0 else "Trung tính"
            explanations.append((token, impact_score, tendency))
        explanations.sort(key=lambda x: abs(x[1]), reverse=True)
        return explanations[:top_k]


# ============================================================
# CÁC HÀM TÍNH CHỈ SỐ ĐÁNH GIÁ
# ============================================================

def calculate_metrics(y_true, y_pred, pos_label=POSITIVE_LABEL, neg_label=NEGATIVE_LABEL):
    """
    Tính toán các chỉ số: Confusion Matrix, Accuracy, Precision, Recall và F1-Score
    """
    TP = TN = FP = FN = 0
    
    # Duyệt và đếm Confusion Matrix
    for t, p in zip(y_true, y_pred):
        if t == pos_label and p == pos_label:
            TP += 1 # True Positive
        elif t == neg_label and p == neg_label:
            TN += 1 # True Negative
        elif t == neg_label and p == pos_label:
            FP += 1 # False Positive
        elif t == pos_label and p == neg_label:
            FN += 1 # False Negative
            
    total = TP + TN + FP + FN
    # Độ chính xác tổng thể
    accuracy = (TP + TN) / total if total > 0 else 0
    
    # Chỉ số của Lớp Dương Tính
    prec_pos = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec_pos = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_pos = 2 * prec_pos * rec_pos / (prec_pos + rec_pos) if (prec_pos + rec_pos) > 0 else 0

    # Chỉ số của Lớp Âm Tính
    prec_neg = TN / (TN + FN) if (TN + FN) > 0 else 0
    rec_neg = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_neg = 2 * prec_neg * rec_neg / (prec_neg + rec_neg) if (prec_neg + rec_neg) > 0 else 0
    
    # Điểm Macro (Điểm trung bình cộng của 2 lớp - công bằng cho dữ liệu mất cân bằng)
    macro_prec = (prec_pos + prec_neg) / 2
    macro_rec = (rec_pos + rec_neg) / 2
    macro_f1 = (f1_pos + f1_neg) / 2
    
    return {
        "accuracy": accuracy,
        "cm": {"TP": TP, "TN": TN, "FP": FP, "FN": FN},
        "pos": {"precision": prec_pos, "recall": rec_pos, "f1": f1_pos},
        "neg": {"precision": prec_neg, "recall": rec_neg, "f1": f1_neg},
        "macro": {"precision": macro_prec, "recall": macro_rec, "f1": macro_f1}
    }

def calculate_roc_auc(y_true, y_proba_pos, pos_label=POSITIVE_LABEL):
    """
    Tính chỉ số ROC-AUC bằng tay (Area Under the Receiver Operating Characteristic Curve).
    y_true: Danh sách nhãn thực tế
    y_proba_pos: Danh sách xác suất dự đoán là lớp Positive (từ 0.0 đến 1.0)
    Sử dụng phương pháp hình thang (Trapezoidal rule)
    """
    # Gom cặp (xác suất, nhãn thực tế) và sắp xếp giảm dần theo xác suất dự đoán
    data = list(zip(y_proba_pos, y_true))
    data.sort(key=lambda x: x[0], reverse=True)
    
    num_pos = sum(1 for p, t in data if t == pos_label)
    num_neg = len(data) - num_pos
    if num_pos == 0 or num_neg == 0:
        return 0.5 # Không thể tính nếu thiếu 1 trong 2 lớp
        
    auc = 0.0
    last_fpr = 0.0
    last_tpr = 0.0
    
    tp = 0
    fp = 0
    
    i = 0
    n = len(data)
    # Duyệt qua các xác suất (xử lý các xác suất bằng nhau thành 1 cụm)
    while i < n:
        current_prob = data[i][0]
        # Xử lý tất cả các mẫu có cùng giá trị xác suất
        while i < n and data[i][0] == current_prob:
            if data[i][1] == pos_label:
                tp += 1
            else:
                fp += 1
            i += 1
            
        # Tính tỷ lệ True Positive và False Positive hiện tại
        tpr = tp / num_pos
        fpr = fp / num_neg
        
        # Cộng dồn diện tích hình thang: Area = dx * (y1 + y2) / 2
        auc += (fpr - last_fpr) * (tpr + last_tpr) / 2.0
        
        last_fpr = fpr
        last_tpr = tpr
        
    return auc


def classify_with_fallback(model, text):
    """Predict with Naive Bayes, then apply lexicon fallback when confidence is low."""
    tokens = tokenize(text)
    token_count = len(tokens)
    known_count = sum(1 for t in tokens if t in model.vocab)
    oov_ratio = 1.0 if token_count == 0 else 1.0 - (known_count / token_count)

    lex_score, lex_pos_hits, lex_neg_hits = lexicon_score(tokens)
    lex_score_adjusted = lex_score
    if token_count <= SHORT_SENTENCE_MAX_TOKENS and lex_score != 0:
        lex_score_adjusted = lex_score * SHORT_SENTENCE_LEXICON_BOOST

    lex_label = None
    if lex_score_adjusted > 0:
        lex_label = POSITIVE_LABEL
    elif lex_score_adjusted < 0:
        lex_label = NEGATIVE_LABEL

    nb_label, log_scores = model.predict(text)
    probs = softmax_from_log_scores(log_scores)
    p_pos = probs.get(POSITIVE_LABEL, 0.0)
    p_neg = probs.get(NEGATIVE_LABEL, 0.0)
    margin = abs(p_pos - p_neg)
    max_prob = max(probs.values()) if probs else 0.0

    uncertain_reasons = []
    if token_count < MIN_VALID_TOKENS:
        uncertain_reasons.append("too_few_tokens")
    if oov_ratio >= OOV_RATIO_THRESHOLD:
        uncertain_reasons.append("high_oov")
    if max_prob < UNCERTAIN_PROB_THRESHOLD:
        uncertain_reasons.append("low_max_prob")
    if margin < UNCERTAIN_MARGIN_THRESHOLD:
        uncertain_reasons.append("low_margin")
    uncertain = len(uncertain_reasons) > 0

    final_label = nb_label
    fallback_used = False
    if uncertain and lex_label:
        final_label = lex_label
        fallback_used = True

    return {
        "final_label": final_label,
        "nb_label": nb_label,
        "log_scores": log_scores,
        "probs": probs,
        "tokens": tokens,
        "token_count": token_count,
        "known_count": known_count,
        "oov_ratio": oov_ratio,
        "lex_score": lex_score,
        "lex_score_adjusted": lex_score_adjusted,
        "lex_pos_hits": lex_pos_hits,
        "lex_neg_hits": lex_neg_hits,
        "uncertain": uncertain,
        "uncertain_reasons": uncertain_reasons,
        "fallback_used": fallback_used,
        "margin": margin,
        "max_prob": max_prob,
    }


# ============================================================
# GIAO DIỆN ỨNG DỤNG (TKINTER)
# ============================================================

class ManualBayesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Naive Bayes - Phân loại cảm xúc văn bản")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        
        self.model = ManualNaiveBayes()
        self.data_X = []
        self.data_y = []
        
        self.csv_path_var = tk.StringVar(value=str(Path(__file__).resolve().parent / DEFAULT_CSV))
        
        self.build_ui()
        self.load_default_if_exists()

    def build_ui(self):
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Arial", 16, "bold"), foreground="#2c3e50")
        style.configure("Heading.TLabel", font=("Arial", 11, "bold"))
        style.configure("TButton", font=("Arial", 10))
        
        title = ttk.Label(self.root, text="MÔ HÌNH PHÂN LOẠI CẢM XÚC BAYES", style="Title.TLabel")
        title.pack(pady=10)
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.tab_data = ttk.Frame(self.notebook, padding=10)
        self.tab_train = ttk.Frame(self.notebook, padding=10)
        self.tab_predict = ttk.Frame(self.notebook, padding=10)
        
        self.notebook.add(self.tab_data, text="1. Dữ liệu")
        self.notebook.add(self.tab_train, text="2. Huấn luyện & Đánh giá")
        self.notebook.add(self.tab_predict, text="3. Dự đoán văn bản")
        
        self.build_data_tab()
        self.build_train_tab()
        self.build_predict_tab()

    # --- TAB DỮ LIỆU ---
    def build_data_tab(self):
        top_frame = ttk.Frame(self.tab_data)
        top_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(top_frame, text="Đường dẫn file CSV:").pack(side=tk.LEFT)
        ttk.Entry(top_frame, textvariable=self.csv_path_var, width=60).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Chọn file", command=self.choose_csv).pack(side=tk.LEFT)
        ttk.Button(top_frame, text="Tải dữ liệu", command=self.load_csv).pack(side=tk.LEFT, padx=5)
        
        self.stats_label = ttk.Label(self.tab_data, text="Chưa có dữ liệu", font=("Arial", 11, "italic"))
        self.stats_label.pack(anchor="w", pady=10)
        
        columns = ("text", "label")
        self.data_tree = ttk.Treeview(self.tab_data, columns=columns, show="headings", height=15)
        self.data_tree.heading("text", text="Văn bản")
        self.data_tree.heading("label", text="Nhãn (Cảm xúc)")
        self.data_tree.column("text", width=700)
        self.data_tree.column("label", width=150, anchor="center")
        
        scroll_y = ttk.Scrollbar(self.tab_data, orient=tk.VERTICAL, command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=scroll_y.set)
        
        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

    # --- TAB HUẤN LUYỆN ---
    def build_train_tab(self):
        control_frame = ttk.Frame(self.tab_train)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Xoá lựa chọn thuật toán, chỉ giữ Multinomial
        ttk.Label(control_frame, text="Thuật toán: Multinomial Naive Bayes", font=("Arial", 10, "italic")).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(control_frame, text="Tỉ lệ tập Test:").pack(side=tk.LEFT, padx=(10, 0))
        self.test_ratio_var = tk.StringVar(value="0.2")
        ttk.Entry(control_frame, textvariable=self.test_ratio_var, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Bắt đầu Huấn luyện & Đánh giá", command=self.run_train).pack(side=tk.LEFT, padx=20)
        
        self.result_text = tk.Text(self.tab_train, wrap=tk.WORD, font=("Consolas", 11))
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=10)
        self.result_text.config(state=tk.DISABLED)

    # --- TAB DỰ ĐOÁN ---
    def build_predict_tab(self):
        ttk.Label(self.tab_predict, text="Nhập văn bản cần phân loại:", style="Heading.TLabel").pack(anchor="w", pady=5)
        
        self.input_text = tk.Text(self.tab_predict, height=5, font=("Arial", 12))
        self.input_text.pack(fill=tk.X)
        self.input_text.insert(tk.END, "The movie had an engaging story, strong performances, and a moving ending.")
        
        btn_frame = ttk.Frame(self.tab_predict)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="Dự đoán", command=self.predict_input).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Lưu mô hình", command=self.save_model).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="Tải mô hình", command=self.load_model).pack(side=tk.LEFT)
        
        self.predict_result_label = ttk.Label(self.tab_predict, text="Kết quả: ", font=("Arial", 14, "bold"), foreground="blue")
        self.predict_result_label.pack(anchor="w", pady=5)
        
        self.predict_details_text = tk.Text(self.tab_predict, wrap=tk.WORD, font=("Consolas", 11))
        self.predict_details_text.pack(fill=tk.BOTH, expand=True)
        self.predict_details_text.config(state=tk.DISABLED)

    # ============================================================
    # CÁC HÀM SỰ KIỆN GIAO DIỆN
    # ============================================================

    def choose_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.csv_path_var.set(path)
            self.load_csv()

    def load_default_if_exists(self):
        if Path(self.csv_path_var.get()).exists():
            self.load_csv()

    def load_csv(self):
        """ Tải file dữ liệu thủ công qua thư viện csv chuẩn """
        path = self.csv_path_var.get()
        if not Path(path).exists():
            messagebox.showerror("Lỗi", "Không tìm thấy file CSV!")
            return
            
        self.data_X.clear()
        self.data_y.clear()
        
        # Đọc file (bắt cả lỗi mã hóa tiếng Việt nếu có)
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or "text" not in reader.fieldnames or "label" not in reader.fieldnames:
                    messagebox.showerror("Lỗi", "File CSV bắt buộc phải có 2 cột tên là: 'text' và 'label'")
                    return
                    
                for row in reader:
                    text = row.get("text", "").strip()
                    if not text: continue
                    
                    try:
                        label = normalize_label(row.get("label", ""))
                        self.data_X.append(text)
                        self.data_y.append(label)
                    except ValueError:
                        pass # Bỏ qua các dòng bị thiếu hoặc sai nhãn
        except Exception as e:
            messagebox.showerror("Lỗi đọc file", str(e))
            return

        self.stats_label.config(text=f"Đã tải xong: {len(self.data_X)} dòng dữ liệu chuẩn.")
        
        # Cập nhật hiển thị (Giới hạn hiển thị 200 dòng để đỡ nặng giao diện)
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
            
        for text, label in zip(self.data_X[:200], self.data_y[:200]): 
            self.data_tree.insert("", tk.END, values=(text, label))

        # Reset lại mô hình khi thay đổi dữ liệu
        self.model = ManualNaiveBayes()
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.config(state=tk.DISABLED)
        
    def run_train(self):
        """ Xử lý việc chia dữ liệu, Huấn luyện và Kiểm tra kết quả """
        if not self.data_X:
            messagebox.showwarning("Cảnh báo", "Bạn chưa tải dữ liệu CSV. Xin mời sang tab Dữ liệu tải trước.")
            return
            
        try:
            test_ratio = float(self.test_ratio_var.get())
            if not (0.01 < test_ratio < 0.99): raise ValueError()
        except ValueError:
            messagebox.showerror("Lỗi", "Tỉ lệ tập Test không hợp lệ (nên điền từ 0.1 tới 0.5)")
            return
            
        # -- TỰ CHIA TẬP TRAIN VÀ TEST (Bằng thuật toán trộn ngẫu nhiên) --
        combined = list(zip(self.data_X, self.data_y))
        random.seed(42) # Cố định seed ngẫu nhiên để kết quả chạy lại luôn giống nhau
        random.shuffle(combined)
        
        shuffled_X, shuffled_y = zip(*combined)
        shuffled_X = list(shuffled_X)
        shuffled_y = list(shuffled_y)
        
        split_idx = int(len(shuffled_X) * (1 - test_ratio))
        
        # Cắt mảng tạo Train set
        X_train = shuffled_X[:split_idx]
        y_train = shuffled_y[:split_idx]
        
        # Cắt mảng tạo Test set
        X_test = shuffled_X[split_idx:]
        y_test = shuffled_y[split_idx:]
        
        # -- HUẤN LUYỆN --
        self.model = ManualNaiveBayes(alpha=1.0)
        self.model.fit(X_train, y_train)
        
        # -- DỰ ĐOÁN VÀ ĐÁNH GIÁ (Trên tập Test) --
        y_pred = []
        y_proba_pos = [] # Lưu xác suất phần trăm của lớp Dương tính để tính ROC-AUC
        
        for text in X_test:
            pred_label, _ = self.model.predict(text)
            probs = self.model.predict_proba(text)
            
            y_pred.append(pred_label)
            # Lấy xác suất lớp POSITIVE_LABEL (nếu có)
            y_proba_pos.append(probs.get(POSITIVE_LABEL, 0.0))
        
        metrics = calculate_metrics(y_test, y_pred)
        roc_auc = calculate_roc_auc(y_test, y_proba_pos, pos_label=POSITIVE_LABEL)
        
        # In báo cáo kết quả
        report = []
        report.append("=========================================================================")
        report.append("          KẾT QUẢ HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH NAIVE BAYES           ")
        report.append("=========================================================================\n")
        report.append(f"- Thuật toán sử dụng        : MULTINOMIAL")
        report.append(f"- Tổng số mẫu (Văn bản) : {len(self.data_X)}")
        report.append(f"- Số mẫu dùng để Huấn Luyện : {len(X_train)}")
        report.append(f"- Số mẫu dùng để Kiểm Tra   : {len(X_test)}")
        report.append(f"- Kích thước bộ Từ Vựng học được: {self.model.vocab_size} từ duy nhất\n")
        
        report.append(f">> ĐỘ CHÍNH XÁC TỔNG THỂ (Accuracy): {metrics['accuracy']*100:.2f}%")
        report.append(f">> ROC-AUC SCORE (Diện tích dưới đường cong): {roc_auc:.4f}\n")
        
        report.append("--- CHỈ SỐ ĐÁNH GIÁ CHI TIẾT ---")
        report.append(f"DƯƠNG TÍNH : Precision = {metrics['pos']['precision']:.4f}  |  Recall = {metrics['pos']['recall']:.4f}  |  F1-Score = {metrics['pos']['f1']:.4f}")
        report.append(f"ÂM TÍNH    : Precision = {metrics['neg']['precision']:.4f}  |  Recall = {metrics['neg']['recall']:.4f}  |  F1-Score = {metrics['neg']['f1']:.4f}")
        report.append(f"TRUNG BÌNH : Precision = {metrics['macro']['precision']:.4f}  |  Recall = {metrics['macro']['recall']:.4f}  |  F1-Score = {metrics['macro']['f1']:.4f}\n")
        
        cm = metrics['cm']
        report.append("--- MA TRẬN NHẦM LẪN (Confusion Matrix) ---")
        report.append("                        Dự đoán ÂM TÍNH     Dự đoán DƯƠNG TÍNH")
        report.append(f"Thực tế ÂM TÍNH           {cm['TN']:7d}               {cm['FP']:7d}")
        report.append(f"Thực tế DƯƠNG TÍNH        {cm['FN']:7d}               {cm['TP']:7d}\n")
        
        report.append("--- CÁC MẪU DỰ ĐOÁN SAI (Lấy tối đa 10 mẫu) ---")
        wrong_count = 0
        for text, true_label, pred_label in zip(X_test, y_test, y_pred):
            if true_label != pred_label:
                wrong_count += 1
                if wrong_count <= 10:
                    report.append(f"[{wrong_count}] Thực tế: {true_label} | Dự đoán: {pred_label}")
                    report.append(f"    Văn bản: {text}\n")
        if wrong_count == 0:
            report.append("Tuyệt vời! Không có mẫu nào dự đoán sai trong lần chạy này.\n")
        else:
            report.append(f"*(Tổng cộng có {wrong_count} mẫu dự đoán sai)*")
        
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "\n".join(report))
        self.result_text.config(state=tk.DISABLED)
        
        messagebox.showinfo("Thành công", "Huấn luyện và Đánh giá đã hoàn tất!\nHãy xem chi tiết tại tab hiện tại hoặc sang tab 3 để thử dự đoán.")

    def predict_input(self):
        """ Dự đoán nhãn cho câu người dùng tự nhập """
        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            return
            
        if not self.model.is_trained:
            # Tự động huấn luyện toàn bộ dữ liệu nếu chưa bấm huấn luyện bên tab 2
            if not self.data_X:
                messagebox.showerror("Lỗi", "Chưa có dữ liệu. Vui lòng quay lại tab 1 để tải file CSV!")
                return
            self.model = ManualNaiveBayes(alpha=1.0)
            self.model.fit(self.data_X, self.data_y)
            messagebox.showinfo("Thông báo", "Vì bạn chưa train mô hình ở tab 2, hệ thống tự động học trên TOÀN BỘ DỮ LIỆU có sẵn với thuật toán Multinomial.")
            
        # Gọi mô hình để dự đoán (kèm fallback theo Package 3)
        result = classify_with_fallback(self.model, text)
        pred_label = result["nb_label"]
        final_label = result["final_label"]
        log_scores = result["log_scores"]
        probs = result["probs"]
        explanations = self.model.explain(text)
        
        # Hiển thị tiêu đề kết quả
        label_text = f"Kết quả dự đoán: {final_label}"
        if result["fallback_used"]:
            label_text += " (lexicon)"
        elif result["uncertain"]:
            label_text += " (không chắc)"
        self.predict_result_label.config(text=label_text)
        
        # Viết chi tiết ra bảng
        details = []
        details.append(f"Kết quả Naive Bayes: {pred_label} | Kết quả cuối: {final_label}")
        details.append("--- PHẦN TRĂM XÁC SUẤT ---")
        for c, p in probs.items():
            details.append(f"{c:15s} : {p*100:6.2f}%")
            
        details.append("\n--- ĐIỂM LOGARIT (Điểm thô trước khi ra phần trăm) ---")
        for c, s in log_scores.items():
            details.append(f"{c:15s} : {s:.4f}")

        details.append("\n--- ĐỘ TIN CẬY & DỮ LIỆU ---")
        details.append(
            f"Token hợp lệ: {result['token_count']} | OOV: {result['oov_ratio']*100:5.1f}% | "
            f"Margin: {result['margin']:.3f} | MaxProb: {result['max_prob']:.3f}"
        )
        if result["uncertain"]:
            reason_map = {
                "too_few_tokens": "quá ít token",
                "high_oov": "nhiều từ lạ",
                "low_max_prob": "xác suất thấp",
                "low_margin": "chênh lệch nhỏ",
            }
            reason_text = ", ".join(reason_map[r] for r in result["uncertain_reasons"])
            details.append(f">> Tín hiệu không chắc: {reason_text}")
        else:
            details.append(">> Tín hiệu ổn định")

        details.append("\n--- LEXICON (TỪ CẢM XÚC) ---")
        details.append(
            f"Điểm lexicon: {result['lex_score']} | Sau boost: {result['lex_score_adjusted']:.2f}"
        )
        details.append(
            f"Từ tích cực: {', '.join(result['lex_pos_hits'][:10]) if result['lex_pos_hits'] else '-'}"
        )
        details.append(
            f"Từ tiêu cực: {', '.join(result['lex_neg_hits'][:10]) if result['lex_neg_hits'] else '-'}"
        )
        if result["fallback_used"]:
            details.append(">> Đã dùng lexicon fallback do độ tin cậy thấp.")
        elif result["uncertain"]:
            details.append(">> Không có fallback phù hợp, giữ kết quả Naive Bayes.")
            
        details.append("\n--- GIẢI THÍCH SỰ ẢNH HƯỞNG CỦA TỪNG TỪ ĐẾN KẾT QUẢ ---")
        details.append("Từ vựng                Điểm ảnh hưởng (Impact)    Từ này nghiêng về")
        details.append("-" * 70)
        
        if not explanations:
            details.append(">> Các từ trong câu này hoàn toàn MỚI, mô hình chưa từng gặp lúc huấn luyện.")
            details.append(">> Nên cả 2 lớp được đánh giá xác suất dựa vào điểm số mặc định.")
        else:
            for word, score, tendency in explanations:
                details.append(f"{word:20s}   {score:>12.6f}             {tendency}")
                
        self.predict_details_text.config(state=tk.NORMAL)
        self.predict_details_text.delete("1.0", tk.END)
        self.predict_details_text.insert(tk.END, "\n".join(details))
        self.predict_details_text.config(state=tk.DISABLED)

    def save_model(self):
        if not self.model.is_trained:
            messagebox.showwarning("Cảnh báo", "Mô hình chưa được huấn luyện. Hãy huấn luyện trước khi lưu!")
            return
            
        path = filedialog.asksaveasfilename(
            title="Lưu mô hình",
            defaultextension=".pkl",
            filetypes=[("Pickle Model", "*.pkl"), ("All files", "*.*")],
            initialfile="manual_bayes_model.pkl"
        )
        if not path: return
        
        try:
            with open(path, "wb") as f:
                pickle.dump(self.model, f)
            messagebox.showinfo("Thành công", f"Đã lưu mô hình thành công tại:\n{path}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu mô hình: {e}")

    def load_model(self):
        path = filedialog.askopenfilename(
            title="Tải mô hình",
            filetypes=[("Pickle Model", "*.pkl"), ("All files", "*.*")]
        )
        if not path: return
        
        try:
            with open(path, "rb") as f:
                loaded_model = pickle.load(f)
                
            if isinstance(loaded_model, ManualNaiveBayes):
                self.model = loaded_model
                messagebox.showinfo("Thành công", "Đã tải mô hình thành công! Bạn có thể chuyển sang Tab 3 để dự đoán.")
            else:
                messagebox.showerror("Lỗi", "File không chứa mô hình ManualNaiveBayes hợp lệ!")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải mô hình: {e}")


# ============================================================
# KHỞI CHẠY CHƯƠNG TRÌNH
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = ManualBayesApp(root)
    root.mainloop()
