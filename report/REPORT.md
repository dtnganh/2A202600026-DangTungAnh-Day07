# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Đặng Tùng Anh
**Nhóm:** 02 - E402
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Hai vector hướng cùng về một hướng trong không gian, biểu thị sự tương đồng cao về mặt ngữ nghĩa (semantic similarity) giữa hai đoạn văn bản, bất chấp sự khác biệt về độ dài hay từ ngữ cụ thể.

**Ví dụ HIGH similarity:**
- Sentence A: The dog is chasing the cat.
- Sentence B: A canine is running after a feline.
- Tại sao tương đồng: Hai câu có cùng ý nghĩa dù sử dụng các từ đồng nghĩa khác nhau.

**Ví dụ LOW similarity:**
- Sentence A: Computers are essential for modern science.
- Sentence B: I prefer to eat organic vegetables.
- Tại sao khác: Hai câu nói về hai chủ đề hoàn toàn khác nhau (công nghệ vs ẩm thực).

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity chỉ quan tâm đến góc giữa các vector, giúp loại bỏ ảnh hưởng của độ dài văn bản (magnitude). Trong NLP, độ dài đoạn văn thường không phản ánh ý nghĩa, nên Cosine similarity cho kết quả ổn định hơn cho các tài liệu có độ dài khác nhau.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Phép tính: `num_chunks = ceil((10,000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11)`
> Đáp án: 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Đáp án: `num_chunks = ceil((10,000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = 25 chunks`. Số lượng chunks tăng lên. Chúng ta muốn overlap nhiều hơn để tránh việc các ý tưởng hoặc thông tin quan trọng bị cắt đứt phũ phàng ở biên của mỗi mảnh, giúp AI duy trì ngữ cảnh tốt hơn khi tìm kiếm.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** VinUni Student Policies & Regulations

**Tại sao nhóm chọn domain này?**
> Tài liệu policy của VinUni có cấu trúc Markdown rõ ràng với header phân cấp (`##`, `###`), mỗi section biểu diễn một điều khoản độc lập — lý tưởng để test retrieval precision. Domain đa ngôn ngữ (en/vi) cũng cho phép test khả năng multilingual embedding của Voyage AI. Ngoài ra, đây là tài liệu thực tế mà sinh viên VinUni cần tra cứu thường ngày, nên benchmark queries có tính ứng dụng cao.

### Data Inventory

| # | Tên tài liệu | Nguồn | Kích thước | Metadata đã gán |
|---|--------------|-------|------------|-----------------|
| 01 | Sexual_Misconduct_Response_Guideline | policy.vinuni.edu.vn | 26.7 KB | category=Guideline, lang=en, dept=SAM, topic=student_life |
| 02 | Admissions_Regulations_GME_Programs | policy.vinuni.edu.vn | 29.0 KB | category=Regulation, lang=en, dept=CHS, topic=academics |
| 03 | Cam_Ket_Chat_Luong_Dao_Tao | policy.vinuni.edu.vn | 43.3 KB | category=Report, lang=vi, dept=University, topic=academics |
| 04 | Chat_Luong_Dao_Tao_Thuc_Te | policy.vinuni.edu.vn | 18.4 KB | category=Report, lang=vi, dept=University, topic=academics |
| 05 | Doi_Ngu_Giang_Vien_Co_Huu | policy.vinuni.edu.vn | 9.9 KB | category=Report, lang=vi, dept=University, topic=academics |
| 06 | English_Language_Requirements | policy.vinuni.edu.vn | 14.0 KB | category=Policy, lang=en, dept=University, topic=academics |
| 07 | Lab_Management_Regulations | policy.vinuni.edu.vn | 46.1 KB | category=Regulation, lang=en, dept=Operations, topic=safety |
| 08 | Library_Access_Services_Policy | policy.vinuni.edu.vn | 3.5 KB | category=Policy, lang=en, dept=Library, topic=student_life |
| 09 | Student_Grade_Appeal_Procedures | policy.vinuni.edu.vn | 6.1 KB | category=SOP, lang=en, dept=AQA, topic=academics |
| 10 | Tieu_Chuan_ANAT_PCCN | policy.vinuni.edu.vn | 4.6 KB | category=Standard, lang=vi, dept=Operations, topic=safety |
| 11 | Quy_Dinh_Xu_Ly_Su_Co_Chay | policy.vinuni.edu.vn | 2.7 KB | category=Regulation, lang=vi, dept=Operations, topic=safety |
| 12 | Scholarship_Maintenance_Criteria | policy.vinuni.edu.vn | 5.7 KB | category=Guideline, lang=en, dept=SAM, topic=finance |
| 13 | Student_Academic_Integrity | policy.vinuni.edu.vn | 41.8 KB | category=Policy, lang=en, dept=AQA, topic=academics |
| 14 | Student_Award_Policy | policy.vinuni.edu.vn | 14.7 KB | category=Policy, lang=en, dept=SAM, topic=student_life |
| 15 | Student_Code_of_Conduct | policy.vinuni.edu.vn | 17.9 KB | category=Policy, lang=en, dept=SAM, topic=student_life |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `category` | string | Guideline / Policy / SOP / Standard / Report | Phân loại loại tài liệu — có thể filter_search theo type |
| `language` | string | `en` / `vi` | Biết ngôn ngữ gốc để chọn embedder phù hợp |
| `department` | string | SAM / AQA / CHS / Operations / Library | Filter theo phòng ban phụ trách |
| `topic` | string | academics / student_life / safety / finance | Filter theo chủ đề để thu hẹp search space |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

So sánh 3 strategy baseline trên file `13_Student_Academic_Integrity.md` (41.8 KB, tài liệu dài nhất, cấu trúc header rõ):

| Strategy | Chunk Count | Avg Length (ký tự) | Preserves Context? |
|----------|-------------|---------------------|-------------------|
| FixedSize(500) | 95 | 500 | Cắt giữa câu/điều khoản |
| SentenceChunker | 66 | ~350 | Giữ câu, nhưng có thể cắt đứt 1 điều khoản |
| RecursiveChunker | 52 | ~480 | Tốt hơn FixedSize, nhưng không hiểu cấu trúc policy |
| **HeaderAwareChunker (của tôi)** | **44** | **~620** | **Mỗi chunk = 1 điều khoản hoàn chỉnh** |

### Strategy Của Tôi

**Loại:** Custom strategy (`HybridChunker`)

**Mô tả cách hoạt động:**
> Chiến lược lai (Hybrid) thực hiện chia tài liệu theo hai bước. Bước 1: Sử dụng `HeaderChunker` để tách văn bản dựa trên các tiêu đề Markdown (#, ##), giúp giữ trọn vẹn logic của từng điều mục. Bước 2: Nếu bất kỳ mục nào sau khi chia vẫn quá dài (vượt quá 1500 ký tự), hệ thống sẽ gọi `RecursiveChunker` để tiếp tục băm nhỏ mảnh đó, đảm bảo an toàn cho giới hạn token của OpenAI API.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Các tài liệu của VinUni (SOP, Biểu mẫu, Quy định) có cấu trúc phân cấp rất rõ ràng. Việc chia theo tiêu đề giúp AI trả lời chính xác từng điều khoản mà không bị lẫn lộn thông tin. Bước chia bổ sung giúp xử lý các phụ lục dài không có tiêu đề con.

**Code snippet (nếu custom):**
```python
class HybridChunker:
    def __init__(self, max_chunk_size: int = 1500):
        self.max_chunk_size = max_chunk_size
        self.header_chunker = HeaderChunker()
        self.recursive_chunker = RecursiveChunker(chunk_size=max_chunk_size)

    def chunk(self, text: str) -> list[str]:
        sections = self.header_chunker.chunk(text)
        final_chunks = []
        for section in sections:
            if len(section) <= self.max_chunk_size:
                final_chunks.append(section)
            else:
                final_chunks.extend(self.recursive_chunker.chunk(section))
        return final_chunks
```

### So Sánh: Strategy của tôi vs Baseline

So sánh trên cùng file `13_Student_Academic_Integrity.md`:

| Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|----------|-------------|------------|-------------------|
| FixedSize(500) — baseline | 95 | 500 ký tự | Thấp — cắt giữa điều khoản |
| RecursiveChunker — baseline | 52 | 480 ký tự | Trung bình |
| **HybridChunker — của tôi** | **~250** | **~600 ký tự** | **Rất cao — bảo toàn cấu trúc + an toàn cho API** |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|-----------------------|-----------|----------|
| **Tôi (Đặng Tùng Anh)** | **HybridChunker** | **9/10** | Cân bằng giữa cấu trúc và giới hạn token | Phức tạp hơn để thiết lập |
| [Mai Tấn Thành] | HeaderAwareChunker | 8/10 | Giữ cấu trúc tuyệt đối | Dễ bị lỗi API nếu section quá dài |
| [Hồ Nhất Khoa] | Semantic Chunking | 9/10 | Tách theo ngữ nghĩa cực kỳ chính xác | Rất chậm và tốn tài nguyên |
| [Nguyễn Đức Hoàng Phúc] | Recursive Chunking | 7/10 | Tin cậy, dễ triển khai, khớp token limit | Bỏ qua cấu trúc logic của tài liệu |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> HybridChunker là tốt nhất vì nó kết hợp được cả tính logic của HeaderChunker và tính an toàn của RecursiveChunker, đảm bảo hệ thống RAG hoạt động ổn định với mọi kích cỡ tài liệu.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng Regex `(?<=[.!?])\s+` (lookbehind) để tách câu dựa trên các dấu kết thúc (.!?). Cách tiếp cận này giúp giữ lại dấu câu ở cuối mỗi câu và chỉ tách khi có khoảng trắng theo sau, tránh lỗi với các chữ viết tắt.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Sử dụng thuật toán đệ quy duyệt qua danh sách các dấu ngăn cách ưu tiên (\n\n, \n, space...). Nếu một đoạn văn vượt quá `chunk_size`, nó sẽ bị xẻ nhỏ tại dấu ngăn cách cao nhất có thể, nếu không tìm thấy dấu ngăn cách phù hợp sẽ cắt theo từng ký tự.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Tài liệu được lưu trữ dưới dạng list các bản ghi (dict) trong bộ nhớ. Khi tìm kiếm, query được chuyển thành vector qua OpenAI API, sau đó duyệt qua toàn bộ kho để tính Cosine Similarity và trả về Top-K kết quả có điểm cao nhất.

**`search_with_filter` + `delete_document`** — approach:
> Thực hiện lọc (Filtering) trước khi tính toán độ tương đồng để tiết kiệm tài nguyên. Phép xóa (`delete_document`) thực hiện bằng cách lọc bỏ tất cả các chunks có `doc_id` tương ứng khỏi danh sách lưu trữ.

### KnowledgeBaseAgent

**`answer`** — approach:
> Xây dựng prompt bao gồm 3 phần: Chỉ thị hệ thống (system prompt), Ngữ cảnh được truy xuất (Retrieved Context) và Câu hỏi của người dùng. Agent được ra lệnh chỉ sử dụng thông tin trong ngữ cảnh để trả lời.

### Test Results

```
tests/test_solution.py::TestDocumentDataclass::test_document_creation PASSED [  2%]
tests/test_solution.py::TestDocumentDataclass::test_document_content_is_required PASSED [  4%]
...
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | The cat sits on the mat. | The feline is resting on the rug. | high | 0.9023 | Yes |
| 2 | I love programming in Python. | I enjoy coding with the Python language. | high | 0.8951 | Yes |
| 3 | The weather is sunny today. | Nuclear physics is a complex subject. | low | 0.0542 | Yes |
| 4 | Artificial Intelligence is transforming the world. | The rapid development of AI is changing society. | high | 0.8812 | Yes |
| 5 | Apples are a nutritious fruit. | Global warming causes rising sea levels. | low | -0.1034 | Yes |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả ở cặp 1 và 2 cho thấy Embedding hiểu được sự tương đồng về khái niệm (feline = cat, coding = programming) mà không cần trùng lặp từ vựng. Điều này chứng tỏ Vector representation biểu diễn nghĩa của từ trong một không gian ngữ nghĩa trừu tượng rất tốt.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer (tóm tắt) | Tài liệu liên quan |
|---|-------|-----------------------|--------------------|
| 1 | What are all the conditions a student must maintain to stay in good academic standing at VinUni? | Duy trì GPA ≥ ngưỡng học bổng; không vi phạm academic integrity; tuân thủ code of conduct; đáp ứng tiêu chí xét giải thưởng | `12_Scholarship` + `13_Academic_Integrity` + `14_Award_Policy` + `15_Code_of_Conduct` |
| 2 | What safety and conduct regulations must students follow when using VinUni campus facilities? | Tuân thủ quy định phòng lab; quy trình xử lý sự cố cháy; quy tắc ứng xử chung; chính sách chống xâm hại tình dục | `07_Lab_Management` + `11_Fire_Safety` + `15_Code_of_Conduct` + `01_Sexual_Misconduct` |
| 3 | What are the admission and language requirements for students entering medical programs at VinUni? | Đáp ứng chuẩn tiếng Anh (IELTS/TOEFL); đạt điểm chuẩn tuyển sinh GME; đáp ứng tiêu chuẩn ANAT PCCN | `02_Admissions_GME` + `06_English_Language` + `10_Tieu_Chuan_ANAT` |
| 4 | What procedures and consequences apply when a student breaks university rules? | Quy trình khiếu nại/kháng nghị; hình thức kỷ luật theo mức độ vi phạm; xử lý gian lận học thuật; xử lý hành vi xâm hại | `01_Sexual_Misconduct` + `09_Grade_Appeal` + `13_Academic_Integrity` + `15_Code_of_Conduct` |
| 5 | How does VinUni evaluate and ensure the quality of its academic programs and teaching staff? | Cam kết chất lượng đào tạo; báo cáo chất lượng thực tế; tiêu chuẩn đội ngũ giảng viên; tiêu chí duy trì học bổng như thước đo kết quả | `03_Cam_Ket_Chat_Luong` + `04_Chat_Luong_Thuc_Te` + `05_Doi_Ngu_Giang_Vien` + `12_Scholarship` |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Good standing conditions | `12_Scholarship...md`: GPA >= 2.5, EXCEL eval | 0.695 | Yes | GPA 2.5, không kỷ luật Tier 3-4, EXCEL |
| 2 | Campus safety/conduct | `15_Student_Code...md`: Prohibit harmful conduct | 0.665 | Yes | Cấm hành vi xúc phạm/đe dọa, giữ liêm chính |
| 3 | Medical admission reqs | `02_Admissions_GME...md`: Eligible candidates | 0.671 | Yes | GPA MD >= 7/10, sức khỏe, đạt chuẩn IELTS |
| 4 | Rule breaking consequences | `13_Student_Academic...md`: Tier 3 Sanctions | 0.618 | Yes | Điều tra -> Báo cáo -> Council kỷ luật |
| 5 | Quality & Staff evaluation | `06_English...md`: Verify documents (Wrong) | 0.610 | No | "I don't know" (Thiếu ngữ cảnh tiếng Việt) |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 4 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Em học được từ các thành viên khác phương pháp chia nhỏ (chunking) tối ưu cho các tài liệu khoa học/báo cáo có nhiều chương mục phức tạp, đảm bảo không làm mất đi các liên kết thông tin giữa các phần. Ngoài ra, cách các bạn gán metadata đa dạng (như topic, category) đã giúp em hiểu rõ tầm quan trọng của việc lọc dữ liệu (filtering) trước khi tìm kiếm để tăng độ chính xác.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Qua buổi demo, em học được cách một số nhóm xử lý data phức tạp bằng cách chuyển đổi chúng về định dạng Markdown hoặc JSON trước khi nhúng, giúp AI hiểu được cấu trúc dữ liệu tốt hơn. Em cũng được biết thêm về một số cách chunking mới như Dynamic Chunking.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Em sẽ đầu tư thêm thời gian vào công đoạn tiền xử lý (cleaning), loại bỏ các ký tự rác và chuẩn hóa định dạng văn bản để vector embedding có chất lượng tốt nhất. Ngoài ra, em muốn thử nghiệm kỹ thuật "Small-to-Big Retrieval" — truy xuất các mảnh nhỏ để tìm kiếm nhanh nhưng gửi ngữ cảnh lớn hơn (parent chunks) cho LLM để câu trả lời có chiều sâu và đầy đủ hơn.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 13 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 7 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 3 / 5 |
| **Tổng** | | **91 / 100** |
