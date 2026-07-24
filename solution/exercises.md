# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng placeholder mặc định bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Với `gpt-4o` và prompt trên: ở 0.0 và 0.5, model đều chọn hang Sơn Đoòng với cấu trúc gần giống nhau (phát hiện 1991, khảo sát 2009, số liệu kích thước) — phản hồi ổn định, ít bất ngờ. Ở 1.0, model chuyển sang chủ đề hoàn toàn khác (rùa Hoàn Kiếm), cho thấy temperature cao hơn làm model "đổi hướng" nhiều hơn. Ở 1.5, quay lại Sơn Đoòng nhưng thêm ẩn dụ sáng tạo hơn (tòa nhà 40 tầng, máy bay Boeing 747) — vẫn đúng sự thật nhưng diễn đạt sinh động và dài hơn.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature khoảng 0.2–0.4. Chatbot hỗ trợ khách hàng cần trả lời nhất quán, chính xác và đáng tin — ví dụ chính sách đổi trả, hướng dẫn sử dụng sản phẩm phải giống nhau mỗi lần hỏi. Temperature thấp giảm nguy cơ model "sáng tạo" ra thông tin sai hoặc mâu thuẫn giữa các phiên chat.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Workload: 10.000 × 3 × 350 = 10,5 triệu token output/ngày. Chi phí output GPT-4o: 10,5M / 1000 × $0,010 ≈ $105/ngày; GPT-4o-mini: 10,5M / 1000 × $0,0006 ≈ $6,3/ngày — GPT-4o đắt hơn khoảng ~17 lần (tỷ lệ giá output $0,010 vs $0,0006). Nên dùng GPT-4o khi xử lý khiếu nại phức tạp, phân tích hợp đồng hoặc tư vấn kỹ thuật cần suy luận sâu. Nên dùng mini cho FAQ đơn giản, tra cứu đơn hàng, chào hỏi — câu trả lời ngắn, mẫu cố định, không cần suy luận phức tạp.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với cùng câu hỏi "Giải thích blockchain là gì?", persona giáo viên tiểu học (~161 từ) dùng ẩn dụ hộp xếp nối tiếp, xưng hô "em", không thuật ngữ kỹ thuật — dễ hiểu như kể chuyện. Persona chuyên gia tài chính (~187 từ) dùng thuật ngữ như DLT, hash, nodes, phi tập trung, bất biến, trình bày theo danh sách đánh số — chuyên sâu và trang trọng hơn. System prompt hoạt động như "chỉ thị đạo diễn": cùng một model nhưng đổi persona là đổi giọng điệu, độ sâu và cách chọn ví dụ, không cần sửa câu hỏi người dùng.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Đoạn văn ~114 từ về Việt Nam: `count_tokens` (tiktoken) = 144 token, ước lượng `số từ / 0.75` = 152 — chênh ~5,6% (heuristic hơi cao hơn tiktoken trong trường hợp này). Tiếng Việt thường tốn nhiều token hơn tiếng Anh cùng độ dài vì BPE/tokenizer của GPT chia từ tiếng Việt thành nhiều subword (dấu thanh, ký tự có dấu, từ ghép dài), trong khi từ tiếng Anh phổ biến thường khớp trọn vẹn với token trong bộ mã hóa đã huấn luyện chủ yếu trên dữ liệu Latin.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi người dùng đang chat trực tiếp (chatbot, trợ lý AI, viết nội dung dài) — token hiện ra ngay giúp giảm cảm giác chờ, người dùng bắt đầu đọc trong khi model vẫn đang sinh text. Non-streaming phù hợp hơn khi ứng dụng cần toàn bộ phản hồi trước khi xử lý tiếp: parse JSON có cấu trúc, gọi API nội bộ, chạy pipeline tự động (batch job, phân loại, trích xuất dữ liệu) — vì code backend cần chuỗi hoàn chỉnh, không cần hiển thị từng chunk cho người dùng.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff (0.1s → 0.2s → 0.4s...) cho server thời gian hồi phục sau lỗi tạm thời thay vì bị dồn dập retry liên tục. Delay tăng dần giảm áp lực lên API đang quá tải và tăng xác suất request sau thành công. Nếu hàng nghìn client cùng retry với delay cố định 1 giây, chúng sẽ đồng loạt gửi lại request sau đúng 1 giây — tạo hiệu ứng "thundering herd" (bầy đàn lao vào), server vừa quá tải lại bị đánh thêm một đợt đồng thời, khó phục hồi.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> **System prompt:** "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt." **"trợ giảng thân thiện"** — đặt vai trò rõ ràng (hỗ trợ học, không phải chatbot chung chung) và giọng điệu gần gũi để sinh viên dễ hỏi tiếp. **"trả lời ngắn gọn bằng tiếng Việt"** — giới hạn độ dài giúp tiết kiệm token/chi phí và phù hợp CLI; chỉ định tiếng Việt tránh model trả lời tiếng Anh khi người dùng hỏi bằng tiếng Việt.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất: **history chỉ giữ 3 lượt** (`history[-6:]`) — sau vài câu hỏi, trợ lý "quên" ngữ cảnh đầu phiên (ví dụ tên sinh viên hoặc chủ đề đã thảo luận). **Cải thiện:** thêm bộ nhớ tóm tắt phiên — mỗi 3 lượt, gọi API một lần (non-streaming) để tóm tắt history thành 2–3 câu, lưu vào biến `session_summary`, rồi gửi kèm system prompt ở mỗi lượt: `"Bối cảnh phiên trước: {session_summary}"`. Như vậy giữ được ngữ cảnh dài hạn mà không gửi toàn bộ history (tiết kiệm token).

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
