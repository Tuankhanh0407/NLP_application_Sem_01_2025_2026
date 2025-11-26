# Trả lời câu hỏi (Bài tập thực hành số 06 - Phần 01 - Giới thiệu về kiến trúc transformer)

## Bài 01: Mô hình ngôn ngữ có mặt nạ

### Câu hỏi 01: Mô hình đã dự đoán đúng từ 'capital' không?

**Trả lời:** Có, mô hình đã dự đoán chính xác từ 'capital' với độ tin cậy rất cao (0.9952). Đây là dự đoán hàng đầu và hoàn toàn chính xác về mặt ngữ nghĩa và địa lý, vì Hà Nội thực sự là thủ đô của Việt Nam.

### Câu hỏi 02: Tại sao các mô hình Encoder-only như BERT lại phù hợp cho tác vụ này?

**Trả lời:** Các mô hình Encoder-only như BERT phù hợp cho tác vụ dự đoán từ bị che khuất vì:

- Chúng sử dụng self-attention hai chiều, cho phép xem xét ngữ cảnh từ cả hai phía (trái và phải) của từ bị che khuất.

- Kiến trúc encoder được thiết kế để hiểu sâu toàn bộ cấu trúc câu và mối quan hệ giữa các từ.

- BERT được tiền huấn luyện chính trên tác vụ dự đoán từ bị che khuất, nên nó được tối ưu hóa đặc biệt cho việc này.

## Bài 02: Dự đoán từ tiếp theo

### Câu hỏi 01: Kết quả sinh ra có hợp lý không?

**Trả lời:** Kết quả sinh ra có phần hợp lý nhưng bị lặp lại nhiều. Đoạn đầu "The best thing about learning NLP is that it's a good way to learn the language" là hợp lý và có ý nghĩa. Tuy nhiên, phần sau bị lặp đi lặp lại cùng một cụm từ nhiều lần, cho thấy hạn chế của mô hình DistilGPT2 trong việc duy trì tính đa dạng và sáng tạo trong văn bản dài.

### Câu hỏi 02: Tại sao các mô hình Decoder-only như GPT lại phù hợp cho tác vụ này?

**Trả lời:** Các mô hình Decoder-only như GPT phù hợp cho tác vụ dự đoán từ tiếp theo vì:

- Chúng được thiết kế để sinh token tiếp theo dựa trên các token đã có trước đó (trái sang phải).

- Sử dụng masked self-attention ngăn không cho mô hình "nhìn trước" các token tương lai.

- GPT được tiền huấn luyện trên tác vụ dự đoán từ tiếp theo, nên nó đã học được phân bố xác suất của ngôn ngữ tự nhiên.

- Phù hợp với bản chất tuần tự của việc sinh văn bản.

## Bài 03: Biểu diễn câu

### Câu hỏi 01: Kích thước (chiều) của vector biểu diễn là bao nhiêu? Con số này tương ứng với tham số nào của mô hình BERT?

**Trả lời:** Kích thước của vector biểu diễn là 768 chiều (`torch.Size([1, 768])`). Con số này tương ứng với tham số `hidden_size` của mô hình DistilBERT. Mỗi mô hình BERT-based có `hidden_size` cố định (BERT-base: 768, BERT-large: 1024) đại diện cho kích thước của không gian vector nơi các biểu diễn ngữ nghĩa được mã hóa.

### Câu hỏi 02: Tại sao chúng ta cần sử dụng `attention_mask` khi thực hiện Mean Pooling?

**Trả lời:** Chúng ta cần sử dụng `attention_mask` khi thực hiện Mean Pooling vì:

- Loại bỏ ảnh hưởng của các token padding (thêm vào để đồng nhất độ dài sequence).

- Đảm bảo chỉ tính trung bình trên các token thực tế có trong câu, không tính các token padding (thường là giá trị 0).

- Giúp vector biểu diễn phản ánh chính xác ngữ nghĩa của câu gốc mà không bị "pha loãng" bởi các giá trị padding.

- Cho phép xử lý nhiều câu cùng lúc với độ dài khác nhau mà vẫn đảm bảo tính toán chính xác.