# PROJECT_PLAN — Kế hoạch dự án VehicleVision

Cập nhật: 2026-10-24 (bỏ phân loại và thống kê hướng di chuyển khỏi phạm vi bắt buộc; thêm lưu ý triển khai). Đây là **kế hoạch**: ngoài `test_yolo.py` chưa có mã nào của hệ thống. Dữ liệu, metrics và thí nghiệm nằm ở [EXPERIMENTS.md](EXPERIMENTS.md). Yêu cầu của môn học ở §8, trạng thái hiện tại ở §9, các quyết định đã duyệt ở §10.

## 1. Mục tiêu và phạm vi

Xây dựng và đánh giá một chương trình dùng YOLO để phát hiện, theo dõi và đếm phương tiện trong video, chạy trên máy cá nhân (Windows 11, RTX 3050 Ti 4 GB), có đủ thông tin để tái lập kết quả.

- **Trong phạm vi:** một video từ một camera cố định, **một đoạn vạch đếm** xác định được vị trí; bốn lớp xe; xuất video và thống kê; đánh giá đếm và tốc độ trên video có ground truth.
- **Tùy chọn (chỉ làm khi phần bắt buộc xong và chủ dự án yêu cầu):** giao diện (Streamlit, cần cài thêm thư viện, phải xin phép), biểu đồ, fine-tune, tối ưu suy luận.
- **Ngoài phạm vi:** nhận dạng biển số, đo tốc độ xe, phát hiện vi phạm, phân loại và thống kê theo hướng di chuyển, nhiều vạch, camera chuyển động, camera trực tiếp, huấn luyện từ đầu, triển khai server, benchmark nhiều mô hình/giải thuật. Đây là **một sản phẩm hoàn chỉnh được kiểm thử và đánh giá**, không phải đề tài so sánh mức độ tối ưu của nhiều giải thuật.

### Thành phẩm cuối cùng

1. **Report tổng kết** mô tả đề tài, cơ sở lý thuyết, thiết kế, triển khai, kiểm thử, thực nghiệm, kết quả, đánh giá và kết luận. Khung dự kiến (tham khảo cách tổ chức của báo cáo mẫu `Đồ_án_tổng_hợp.pdf`, không sao chép nội dung; có thể chỉnh theo yêu cầu của giảng viên):
   1. Giới thiệu đề tài
   2. Cơ sở lý thuyết (object detection/YOLO, multi-object tracking/ByteTrack, hình học kiểm tra vượt vạch)
   3. Các công trình/giải pháp liên quan (chỉ là nội dung tổng quan trong báo cáo, không phải chức năng phải lập trình)
   4. Thiết kế và triển khai hệ thống
   5. Thực nghiệm, kết quả và đánh giá
   6. Thảo luận về hiệu quả và hạn chế
   7. Kết luận và hướng phát triển
2. **GitHub repository hoàn chỉnh:** mã nguồn chạy được, cấu hình, hướng dẫn sử dụng (README), unit test, và tài nguyên minh chứng cần thiết (ví dụ ground truth nhỏ, bảng kết quả, ảnh/clip minh họa ngắn khi quyền sử dụng cho phép). Video gốc, dữ liệu lớn và trọng số vẫn không đưa lên Git.

Tiến độ được quản lý theo **mốc chức năng** (§7) cho tới khi hệ thống hoàn chỉnh, sau đó tổng hợp thành Report.

## 2. Chức năng bắt buộc

1. Đọc video MP4 và phát hiện xe (`car`, `motorcycle`, `bus`, `truck`).
2. Theo dõi: gán ID ổn định cho từng xe.
3. Đếm xe thực sự vượt đoạn vạch ảo theo **ID và loại xe**, hạn chế đếm trùng. Lượt vượt từ cả hai phía của vạch cùng vào một tổng; không phân nhóm và không xuất thông tin hướng.
4. Xuất video chú thích (khung bao, ID, vạch, bộ đếm), `tracks.csv`, `events.csv`, `counts.json`.
5. Đánh giá sai số đếm (so với đếm thủ công) và FPS toàn pipeline.

## 3. Pipeline

```text
video MP4 -> Detection + Tracking (YOLO26n + ByteTrack) -> Counting (vạch ảo) -> Output
                                                                          |
              Evaluation (chạy riêng): counts/events  +  ground truth  ----+
```

Mỗi khung hình: ảnh -> danh sách track (ID, khung bao, lớp, độ tin cậy) -> các sự kiện vượt vạch -> vẽ lên video và ghi `tracks.csv`/`events.csv`; cuối video ghi `counts.json`.

## 4. Vai trò của các tệp mã nguồn dự kiến

Chỉ tạo tệp khi bắt đầu làm chức năng tương ứng; không tạo tệp rỗng hay lớp bọc Ultralytics không cần thiết.

| Tệp | Vai trò |
| --- | --- |
| `test_yolo.py` | Đã có. Thử YOLO26n trên ảnh mẫu. Giữ nguyên. |
| `main.py` | Chạy pipeline: đọc video, gọi `model.track(...)`, gọi bộ đếm, vẽ, ghi video/CSV/JSON. Tham số (video, mô hình, `conf`, tracker, tọa độ vạch...) qua dòng lệnh; cách chọn cuối cùng chốt khi viết mã. |
| `counter.py` | Logic đếm thuần Python/NumPy, **không import YOLO/OpenCV**, để kiểm thử bằng tọa độ tổng hợp. |
| `evaluate.py` | Đọc `events.csv`/`counts.json` và ground truth, tính MAE/MAPE/ghép sự kiện (xem EXPERIMENTS). Đo FPS có thể nằm trong `main.py`. |
| `tests/test_counter.py` | Unit test cho `counter.py` (cần `pytest`, xin phép cài). |
| `data/`, `outputs/` | Video, ground truth lớn, kết quả. Không đưa video/kết quả lên Git. |

## 5. Quyết định kỹ thuật

- **Baseline:** `yolo26n.pt` pretrained COCO; lớp COCO 2, 3, 5, 7 (như `test_yolo.py`); `conf=0.35` (giá trị đang dùng, chưa được chứng minh tối ưu).
- **Tracker chọn tường minh: `tracker="bytetrack.yaml"`.** Tracker mặc định của Ultralytics 8.4.182 là `tracktrack.yaml` (đã đọc `default.yaml` tại tag v8.4.182 và bản trong `.venv`), nên **không gọi ByteTrack là mặc định** và không dựa vào giá trị mặc định. Mỗi lần chạy phải ghi tên tracker, phiên bản Ultralytics và nội dung cấu hình tracker thực tế (`bytetrack.yaml` ở bản này có `track_high_thresh`, `track_low_thresh`, `new_track_thresh`, `track_buffer`, `match_thresh`, `fuse_score`).
- **Dùng thẳng `model.track(...)`** của Ultralytics (detection + tracking một lời gọi); đánh đổi: khó đo riêng thời gian detection và tracking.
- **Điểm tham chiếu** của xe: giữa cạnh dưới khung bao `((x1+x2)/2, y2)`; phương án so sánh: tâm khung bao.
- **Vạch là đoạn thẳng hữu hạn** `a`–`b`; phía dương là phía có `cross(b-a, p-a) > 0` (quy ước nội bộ chỉ để phát hiện đổi phía, không đưa ra đầu ra); vạch suy biến (`a == b`) là lỗi cấu hình.
- **Đếm theo ID:** mỗi ID tối đa một sự kiện. Một track ID **không** bằng một phương tiện thật: một xe có hai ID có thể đếm hai lần, hai xe chung một ID chỉ đếm một lần. Giả định ID không bị tái sử dụng trong một lần chạy chưa kiểm chứng (kiểm tra bằng `tracks.csv`).
- **Đối chiếu được:** `grand_total` (số sự kiện vượt vạch của video) = số dòng dữ liệu `events.csv` = tổng `by_class`. `events.csv` và `counts.json` không có trường hướng (`direction`, `by_direction`). `events.csv` là nguồn gốc, `counts.json` suy ra từ cùng danh sách sự kiện. `tracks.csv` cho phép tra từng ID trên video.
- **Ghi lại tham số vạch** (`a`, `b`, `margin_px`, `max_gap_frames`) trong `counts.json`.

### Quy tắc vượt vạch (để viết test)

Với mỗi ID, giữ `last_side` (phía khác 0 gần nhất) và `last_point`. Gọi `d(p)` là khoảng cách có dấu tới đường thẳng qua `a`, `b`; `side = +1` nếu `d > margin_px`, `-1` nếu `d < -margin_px`, `0` nếu nằm trong vùng đệm.

1. `side == 0` (chạm hoặc trong vùng đệm): không làm gì.
2. Chưa có `last_side`: ghi nhận, chưa đếm.
3. `side` ngược dấu `last_side`: nếu đoạn `last_point -> p` **cắt đoạn vạch hữu hạn** và ID chưa đếm thì phát một sự kiện (đổi từ phía nào sang phía nào đều đếm như nhau, không ghi lại hướng), rồi cập nhật trạng thái. Đi vòng qua đầu mút vạch: không đếm nhưng vẫn cập nhật.
4. Vắng mặt ≤ `max_gap_frames` rồi xuất hiện lại cùng ID: vẫn xét nối hai điểm; vắng lâu hơn: xóa trạng thái.
5. Mỗi ID một sự kiện; vượt lại về sau chỉ ghi nhật ký (`recrossings`).
6. Lớp của sự kiện = lớp xuất hiện nhiều nhất trong lịch sử track tới lúc vượt; hòa thì lấy lớp gần nhất.

| Tình huống | Hệ quả |
| --- | --- |
| Track mất trước vạch | đếm thiếu |
| Đổi ID trước hoặc sau vạch | vẫn đếm đúng 1 lần |
| Đổi ID ngay lúc vượt vạch | có thể đếm thừa hoặc thiếu (giới hạn đã biết; đo bằng mức sự kiện) |
| Hai xe che nhau chung một ID | đếm thiếu |
| Đổi lớp giữa chừng | không đổi tổng, có thể sai lớp |

**Các nhóm test cần có** (tọa độ tổng hợp, không cần YOLO/GPU): vượt từ phía dương sang phía âm và ngược lại (mỗi trường hợp đúng một sự kiện, cùng vào một tổng, không có trường hướng); không tới vạch; chạm vạch rồi quay lại; đi vòng đầu mút; rung trong vùng đệm; vượt rồi quay lại (vẫn 1 sự kiện); mất track ngắn so với dài hơn `max_gap_frames`; đổi ID trước vạch; đổi lớp; nhiều xe cùng khung; vạch suy biến; bất biến đầu ra (`grand_total` = số dòng `events.csv` = tổng `by_class`). Với đổi ID ngay lúc vượt vạch, test chỉ ghi lại hành vi thực tế như giới hạn đã biết, không coi là đúng.

### Lưu ý triển khai cho các Coding Task

Rút ra từ việc đọc dự án tham khảo `codewithaarohi/Vehicle-detection-and-tracking-classwise-using-YOLO11` (chỉ lấy ý tưởng, không dùng mã hay video; xem EXPERIMENTS §9). Đây là ghi chú cho lúc viết mã, không đổi phạm vi hay tiêu chí.

1. **Detection:** truyền `classes=[2, 3, 5, 7]` thẳng vào `model.track(...)`, nhất quán với `test_yolo.py`.
2. **Tracking, khung không có track:** khi `boxes.id is None` thì coi là danh sách track rỗng, không gọi `.int()` trên `None`; vẫn vẽ vạch và bộ đếm, vẫn ghi khung và vẫn gọi bộ đếm với danh sách rỗng để xử lý vắng mặt. Kiểm chứng trên video có đoạn không có xe hoặc `conf` rất cao trên một đoạn ngắn.
3. **Tracking, tách trạng thái giữa các video:** nếu đọc từng khung và gọi `model.track(..., persist=True)`, tracker giữ trạng thái cả khi sang video khác trong cùng tiến trình (Ultralytics 8.4.182), nên dùng đối tượng `YOLO` mới (hoặc tiến trình riêng) cho mỗi video. Bỏ `persist=True` khi gọi từng khung thì ID bị đặt lại mỗi khung. Chọn giữa vòng lặp từng khung và `source=<đường dẫn video>` kèm `stream=True` lúc viết `main.py`, ghi lý do vào nhật ký.
4. **Đầu ra video:** nhãn `ID + lớp`, chấm điểm tham chiếu, bộ đếm theo lớp ở góc khung; vẽ vạch và bộ đếm ở **mọi** khung, kể cả khung trống.
5. **Giấy phép tham khảo:** repo mẫu không có `LICENSE`; không chép mã, không dùng video của họ, chỉ ghi vào danh sách nguồn (AC-9).

## 6. Tiêu chí hoàn thành (kiểm chứng được)

Xác nhận ở tuần cuối, trên **tập test** (không phải video dev). Mốc từng tuần ở §7 chỉ là mốc nội bộ trên video dev, không phải nghiệm thu. Hai ngưỡng số là **mục tiêu tham chiếu, chưa đo**; sau baseline có thể điều chỉnh nhưng phải ghi lý do, và kết quả không đạt vẫn phải báo cáo trung thực.

| Mã | Tiêu chí |
| --- | --- |
| AC-1 | Dựng lại môi trường theo README và chạy `test_yolo.py` thành công (chủ dự án xác nhận). |
| AC-2 | Chạy detection hết ≥ 3 video test, xuất video chú thích có số khung bằng video gốc. |
| AC-3 | Video hiển thị ID **và** có `tracks.csv` (đủ cột, `track_id` khớp video). Video có ID một mình không đủ. |
| AC-4 | Unit test logic đếm đạt các nhóm test ở §5. |
| AC-5 | `tracks.csv`, `events.csv`, `counts.json` đúng cấu trúc (không có trường hướng) và thỏa quy tắc đối chiếu `grand_total`. |
| AC-6 | Ground truth thủ công cho tập test (≥ 3 video, ≥ 100 lượt vượt vạch, không tính video dev); báo cáo MAE, MBE, MAPE (không tính mẫu G=0, báo số mẫu hợp lệ) và precision/recall mức sự kiện, kèm sai số từng video. Mục tiêu tham chiếu: MAPE ≤ 15 % trên video điều kiện tốt. |
| AC-7 | Đo FPS toàn pipeline đúng quy trình EXPERIMENTS §5 trên RTX 3050 Ti. Mục tiêu tham chiếu: ≥ 10 FPS với `yolo26n`, đầu vào ≤ 1280x720. |
| AC-8 | Ít nhất hai thí nghiệm phân tích độ nhạy cấu hình (E2 và một trong E1/E4; không phải benchmark nhiều mô hình) trên cùng tập test, mỗi thí nghiệm đổi một biến, có nhật ký tái lập. |
| AC-9 | Tài liệu nêu giới hạn đã quan sát, nguồn bên thứ ba, giấy phép đã xác minh/chưa xác minh. |

## 7. Lộ trình theo mốc chức năng (ước lượng 5–7 tuần)

Thứ tự: Detection → Tracking → Counting → Outputs → Testing & Evaluation → Documentation & Report. Số tuần chỉ là ước lượng; tiến độ đo bằng mốc chức năng, không chia theo đợt slide/demo giữa kỳ, cuối kỳ. Dữ liệu và ground truth làm sớm vì không cần mã tracking.

| Tuần | Việc chính | Mốc nội bộ (trên video dev) |
| --- | --- | --- |
| 1 | Quay/chọn video dev và toàn bộ video test, ghi nguồn + giấy phép; chốt vạch từng video; chốt giao thức ground truth; quyết định về `lap`. | Có video dev; danh sách video test hợp lệ về giấy phép. |
| 2 | `main.py`: detection trên video MP4, xuất video có khung bao. Đếm ground truth lần 1 (video dev, một video test). | Chạy hết video dev, đủ số khung. |
| 3 | Tracking với `bytetrack.yaml`, hiển thị ID, ghi `tracks.csv`. Ground truth lần 1 cho các video test còn lại. | Có video với ID và `tracks.csv`. |
| 4 | `counter.py` + `tests/test_counter.py`; xuất `events.csv`/`counts.json`. Ground truth lần 2 (cách lần 1 ≥ 1 ngày). | Test chạy được; end-to-end ra đủ 3 đầu ra. |
| 5 | `evaluate.py`, đo FPS; **chạy baseline trên tập test.** | Có bảng baseline thật. Đây là mốc tối thiểu có thể nộp. |
| 6 | Thí nghiệm E2 + một trong E1/E4; phân tích lỗi (thiếu/thừa, đổi ID). | Kết quả trên toàn tập test, có nhật ký. |
| 7 | Viết Report tổng kết theo khung §1, hoàn thiện repository (README, hướng dẫn chạy, minh chứng), chạy lại để tái lập, rà soát giấy phép, nghiệm thu AC-1…AC-9. | Bảng nghiệm thu điền trung thực; Report và repository hoàn chỉnh. |

**Phương án 5 tuần:** tuần 1 = tuần 1–2 ở trên (dữ liệu, giao thức ground truth, quyết định `lap`, detection trên video); tuần 2 = tracking và bắt đầu logic đếm; tuần 3 = hoàn thiện đếm + unit test, hoàn tất ground truth (đếm hai lần), chạy baseline; tuần 4 = thí nghiệm (chỉ E2 và một trong E1/E4) và phân tích lỗi; tuần 5 = báo cáo, tái lập, nghiệm thu. Cắt: giao diện, fine-tune, E3–E5, đánh giá detection mức khung hình. Không cắt: unit test đếm, ground truth đếm hai lần, baseline, AC-8.

**Kiểm soát tiến độ:** cuối mỗi tuần ghi ngắn việc đã xong/trễ. Điểm quyết định: cuối tuần 2 (detection video chạy?), cuối tuần 3 (tracking chạy?), cuối tuần 5 (có baseline?); trễ quá 3 ngày ở một điểm thì chuyển sang phương án 5 tuần và ghi phần bị cắt vào báo cáo như giới hạn.

**Blocker thường gặp:**
- `lap` không cài/chạy được: chẩn đoán trước trong một môi trường ảo tạm ngoài `.venv`, xin phép chủ dự án trước khi cài; nếu vẫn không được, hỏi chủ dự án về môi trường Python khác; phương án cuối là tracker IoU đơn giản tự viết (ghi rõ là phần tự làm). Không tự thay `lap` bằng `scipy` khi chưa kiểm chứng tương thích. Chưa kiểm tra `tracktrack` có cần `lap` hay không.
- Thiếu video test hợp lệ: dùng video tự quay (ngắn hơn cũng được, giữ tổng ≥ 100 lượt).
- GPU 4 GB không đủ: xem EXPERIMENTS §6.
- Tracker đổi ID nhiều: ghi nhận và đo, không coi là lỗi dự án.

**Lịch môn học:** lộ trình này không chia theo các đợt báo cáo giữa kỳ/cuối kỳ (quyết định 2026-10-24). Nếu hạn nộp thật ngắn hơn ước lượng thì dùng phương án 5 tuần ở trên. Những điểm khác với quy định môn học được ghi ở §8 để chủ dự án xác nhận với giảng viên.

## 8. Yêu cầu của môn học

Nguồn: `plan-SV.pdf` (kế hoạch môn Đồ án tổng hợp, lưu cục bộ ở thư mục cha, ngoài Git; đã đọc ngày 2026-10-23).

- Mục tiêu môn: thực hiện một project định hướng AI, áp dụng kiến thức đã học (lập trình hàm, cấu trúc dữ liệu và giải thuật, công nghệ phần mềm...).
- Đánh giá: báo cáo giữa kỳ 20 %, cuối kỳ 80 %. **Mỗi đợt cần có Slide, Demo (mã nguồn) và Report.**
- Lịch: 11 tuần (tuần 39–50): tuần 1 chọn đề tài, tuần 2 tìm hiểu tài liệu và xác định mục tiêu, tuần 3–4 hiện thực, **tuần 5 báo cáo giữa kỳ**, tuần 6–9 hiện thực, **tuần 10 báo cáo cuối kỳ**, tuần 11 dự phòng. Góp ý hàng tuần.
- Đề tài được tự đề xuất. Bắt buộc: (1) trình bày tổng thể bài toán; (2) trình bày **ít nhất 3 giải thuật/công trình khoa học liên quan**; (3) mô tả đầu vào, đầu ra và **chạy code demo**.
- Môn quy định nhóm 3–4 sinh viên; dự án này làm cá nhân (theo quyết định của chủ dự án).

**Phạm vi công việc chủ dự án giao (2026-10-24)** — không thay đổi văn bản quy định môn học:
- Thành phẩm là **một Report tổng kết** và **một GitHub repository hoàn chỉnh** (§1). Roadmap không được điều phối theo Slide + Demo + Report của từng đợt.
- Yêu cầu "ít nhất 3 giải thuật/công trình liên quan" được đáp ứng bằng **chương 3 của Report** (tổng quan, dự kiến: họ YOLO, ByteTrack, và một phương pháp tracking cơ sở như SORT; nguồn cần đối chiếu trước khi trích, xem EXPERIMENTS §9). Không lập trình, huấn luyện hay so sánh ba thuật toán.
- Yêu cầu mô tả đầu vào/đầu ra và chứng minh chương trình chạy được **giữ nguyên**: `main.py` chạy trên video thật, video kết quả và các tệp thống kê là minh chứng.
- Áp dụng kiến thức môn học: giải thuật và cấu trúc dữ liệu ở logic đếm (hình học vector, trạng thái theo ID), công nghệ phần mềm ở unit test và Git.

**Điểm khác với quy định môn học, cần chủ dự án xác nhận với giảng viên** (chưa được xác nhận là đã được miễn):
1. `plan-SV.pdf` yêu cầu Slide, Demo và Report ở **cả hai đợt** giữa kỳ (20 %) và cuối kỳ (80 %); kế hoạch dự án chỉ lập cho một Report tổng kết và repository.
2. `plan-SV.pdf` quy định nhóm 3–4 sinh viên; dự án làm cá nhân.

## 9. Trạng thái hiện tại (cập nhật 2026-10-23)

| Hạng mục | Trạng thái | Bằng chứng |
| --- | --- | --- |
| Môi trường: Python 3.14.0, PyTorch 2.13.0+cu132, torchvision 0.28.0+cu132, Ultralytics 8.4.182, opencv-python 5.0.0.93, NumPy 2.5.2 | Đã kiểm chứng | `pip list`, `pip check` không lỗi; `torch.cuda.is_available()` = True trên RTX 3050 Ti (2026-10-23) |
| YOLO26n trên ảnh mẫu (`test_yolo.py`) | Đã chạy | Ảnh kết quả trong `runs/` (cục bộ) |
| Tài liệu: README, PROJECT_PLAN, EXPERIMENTS, requirements | Đã commit | `f4f537b` trên `main` |
| Detection trên video (`main.py`) | Chưa làm; là nhiệm vụ tiếp theo | — |
| Tracking (ByteTrack), đếm, xuất CSV/JSON, unit test, đánh giá | Chưa làm | — |
| Video dev/test, ground truth | Chưa có | — |
| `lap`, `pytest`, `scipy` | Chưa cài | `pip list` |
| Giấy phép dự án | Chưa chọn, chưa có `LICENSE` | — |

## 10. Quyết định đã duyệt

| Ngày | Quyết định |
| --- | --- |
| 2026-10-19 | Đề tài: phát hiện, theo dõi và đếm xe trong video với YOLO + OpenCV; một camera cố định, một vạch đếm; làm cá nhân, 5–7 tuần. |
| 2026-10-19 | Không tự cài/nâng cấp thư viện; không tự commit/push; không đưa video, dữ liệu, trọng số lên Git; không tự gán giấy phép MIT. |
| 2026-10-19 | Baseline YOLO26n pretrained, 4 lớp COCO, `conf=0.35`; tracker chọn tường minh `bytetrack.yaml` (mặc định của thư viện là `tracktrack.yaml`). |
| 2026-10-20 | **Tinh giản dự án và chuyển sang triển khai**: tài liệu chỉ gồm README, PROJECT_PLAN, EXPERIMENTS (+ requirements); không tái tạo bộ 10 tệp cũ (bản sao lưu nằm ngoài repo). Cấu trúc mã mục tiêu: `main.py`, `counter.py`, `evaluate.py`, `tests/test_counter.py`; chỉ tạo tệp khi làm chức năng đó. |
| 2026-10-20 | Thứ tự lập trình: detection video → tracking ByteTrack → đếm vạch → xuất video/CSV/JSON → unit test → đánh giá sai số và FPS → giao diện chỉ khi phần lõi xong. |
| 2026-10-20 | Repository nộp giảng viên không chứa tệp hướng dẫn riêng cho Claude Code: dùng `CLAUDE.local.md`, loại trừ qua `.git/info/exclude`. |
| 2026-10-24 | **Thành phẩm cuối: một Report tổng kết (khung 7 chương ở §1) và một GitHub repository hoàn chỉnh.** VehicleVision là sản phẩm hoàn chỉnh được kiểm thử và đánh giá, không phải đề tài so sánh giải thuật. |
| 2026-10-24 | Tiến độ quản lý theo mốc chức năng (Detection → Tracking → Counting → Outputs → Testing & Evaluation → Documentation & Report), không theo đợt slide/demo giữa kỳ, cuối kỳ. Các công trình liên quan chỉ là nội dung chương 3 của Report. Không tự mở rộng sang benchmark nhiều mô hình, fine-tuning, GUI. |
