BTL Bayes - Demo đầy đủ chức năng
==================================================================

File chính:
- bayes_full_experiment_app.py

Dữ liệu:
- du_lieu_nhan_xet_phim_bayes.csv

Lưu ý: File CSV chỉ giả định dữ liệu tiếng Anh. Nếu dữ liệu là ngôn ngữ khác,
kết quả nhận dạng và đánh giá có thể không chính xác.

Cách chạy:
1. Mở CMD/Terminal tại thư mục này.
2. Cài thư viện:
   pip install -r requirements.txt
3. Chạy:
   python naive_bayes_manual_app.py

Các chức năng trong bản này:
- Tải dữ liệu CSV.
- Xem thống kê dữ liệu.
- Xem trước dữ liệu.
- Chạy GridSearchCV.
- Chạy 5-fold Stratified Cross Validation.
- Chia train/test và đánh giá trên tập test.
- So sánh MultinomialNB, BernoulliNB, GaussianNB và Logistic Regression.
- Hiển thị Accuracy, Precision, Recall, F1-score, ROC-AUC.
- Hiển thị Confusion Matrix.
- Hiển thị Classification Report.
- Hiển thị mẫu dự đoán sai.
- Dự đoán văn bản mới.
- Hiển thị xác suất dự đoán.
- Giải thích từ/cụm từ ảnh hưởng tới dự đoán.
- Lưu mô hình tốt nhất ra file .joblib.
- Tải lại mô hình .joblib.
- Xuất bảng kết quả thực nghiệm ra CSV.
- Bảng kết quả có thanh cuộn ngang/dọc, cột không bị ép nhỏ.

Bản này đã bỏ các tab giới thiệu Chương 1, lý thuyết Chương 2 và tổng kết Chương 4.
Chỉ giữ lại chức năng cần cho demo và thực nghiệm.
