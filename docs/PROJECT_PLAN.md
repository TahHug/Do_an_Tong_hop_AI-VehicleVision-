# PROJECT_PLAN — Kế hoạch dự án VehicleVision

Cập nhật: 2026-10-20. Đây là **kế hoạch**: ngoài `test_yolo.py` chưa có mã nào của hệ thống. Dữ liệu, metrics và thí nghiệm nằm ở [EXPERIMENTS.md](EXPERIMENTS.md).

## 1. Mục tiêu và phạm vi

Xây dựng và đánh giá một chương trình dùng YOLO để phát hiện, theo dõi và đếm phương tiện trong video, chạy trên máy cá nhân (Windows 11, RTX 3050 Ti 4 GB), có đủ thông tin để tái lập kết quả.

- **Trong phạm vi:** một video từ một camera cố định, **một đoạn vạch đếm** xác định được vị trí; bốn lớp xe; xuất video và thống kê; đánh giá đếm và tốc độ trên video có ground truth.
- **Tùy chọn (chỉ khi phần bắt buộc xong):** giao diện (Streamlit, cần cài thêm thư viện, phải xin phép), biểu đồ, fine-tune, tối ưu suy luận.
- **Ngoài phạm vi:** nhận dạng biển số, đo tốc độ xe, phát hiện vi phạm, nhiều vạch, camera chuyển động, camera trực tiếp, huấn luyện từ đầu, triển khai server.

## 2. Chức năng bắt buộc

1. Đọc video MP4 và phát hiện xe (`car`, `motorcycle`, `bus`, `truck`).
2. Theo dõi: gán ID ổn định cho từng xe.
3. Đếm xe vượt vạch ảo theo **ID, hướng và loại xe**, hạn chế đếm trùng.
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
- **Vạch là đoạn thẳng hữu hạn** `a`–`b`; hướng dương là phía có `cross(b-a, p-a) > 0`; vạch suy biến (`a == b`) là lỗi cấu hình.
- **Đếm theo ID:** mỗi ID tối đa một sự kiện. Một track ID **không** bằng một phương tiện thật: một xe có hai ID có thể đếm hai lần, hai xe chung một ID chỉ đếm một lần. Giả định ID không bị tái sử dụng trong một lần chạy chưa kiểm chứng (kiểm tra bằng `tracks.csv`).
- **Đối chiếu được:** `grand_total` (số sự kiện vượt vạch của video) = số dòng dữ liệu `events.csv` = tổng `by_class` = tổng `by_direction`. `events.csv` là nguồn gốc, `counts.json` suy ra từ cùng danh sách sự kiện. `tracks.csv` cho phép tra từng ID trên video.
- **Ghi lại tham số vạch** (`a`, `b`, `margin_px`, `max_gap_frames`) trong `counts.json`.

### Quy tắc vượt vạch (để viết test)

Với mỗi ID, giữ `last_side` (phía khác 0 gần nhất) và `last_point`. Gọi `d(p)` là khoảng cách có dấu tới đường thẳng qua `a`, `b`; `side = +1` nếu `d > margin_px`, `-1` nếu `d < -margin_px`, `0` nếu nằm trong vùng đệm.

1. `side == 0` (chạm hoặc trong vùng đệm): không làm gì.
2. Chưa có `last_side`: ghi nhận, chưa đếm.
3. `side` ngược dấu `last_side`: nếu đoạn `last_point -> p` **cắt đoạn vạch hữu hạn** và ID chưa đếm thì phát một sự kiện (hướng theo chiều đổi dấu), rồi cập nhật trạng thái. Đi vòng qua đầu mút vạch: không đếm nhưng vẫn cập nhật.
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

**Các nhóm test cần có** (tọa độ tổng hợp, không cần YOLO/GPU): vượt hai chiều; không tới vạch; chạm vạch rồi quay lại; đi vòng đầu mút; rung trong vùng đệm; vượt rồi quay lại (vẫn 1 sự kiện); mất track ngắn so với dài hơn `max_gap_frames`; đổi ID trước vạch; đổi lớp; nhiều xe cùng khung; vạch suy biến; bất biến đầu ra (`grand_total` = số dòng `events.csv` = tổng `by_class` = tổng `by_direction`). Với đổi ID ngay lúc vượt vạch, test chỉ ghi lại hành vi thực tế như giới hạn đã biết, không coi là đúng.

## 6. Tiêu chí hoàn thành (kiểm chứng được)

Xác nhận ở tuần cuối, trên **tập test** (không phải video dev). Mốc từng tuần ở §7 chỉ là mốc nội bộ trên video dev, không phải nghiệm thu. Hai ngưỡng số là **mục tiêu tham chiếu, chưa đo**; sau baseline có thể điều chỉnh nhưng phải ghi lý do, và kết quả không đạt vẫn phải báo cáo trung thực.

| Mã | Tiêu chí |
| --- | --- |
| AC-1 | Dựng lại môi trường theo README và chạy `test_yolo.py` thành công (chủ dự án xác nhận). |
| AC-2 | Chạy detection hết ≥ 3 video test, xuất video chú thích có số khung bằng video gốc. |
| AC-3 | Video hiển thị ID **và** có `tracks.csv` (đủ cột, `track_id` khớp video). Video có ID một mình không đủ. |
| AC-4 | Unit test logic đếm đạt các nhóm test ở §5. |
| AC-5 | `tracks.csv`, `events.csv`, `counts.json` đúng cấu trúc và thỏa quy tắc đối chiếu `grand_total`. |
| AC-6 | Ground truth thủ công cho tập test (≥ 3 video, ≥ 100 lượt vượt vạch, không tính video dev); báo cáo MAE, MBE, MAPE (không tính mẫu G=0, báo số mẫu hợp lệ) và precision/recall mức sự kiện, kèm sai số từng video. Mục tiêu tham chiếu: MAPE ≤ 15 % trên video điều kiện tốt. |
| AC-7 | Đo FPS toàn pipeline đúng quy trình EXPERIMENTS §5 trên RTX 3050 Ti. Mục tiêu tham chiếu: ≥ 10 FPS với `yolo26n`, đầu vào ≤ 1280x720. |
| AC-8 | Ít nhất hai thí nghiệm so sánh (E2 và một trong E1/E4) trên cùng tập test, mỗi thí nghiệm đổi một biến, có nhật ký tái lập. |
| AC-9 | Tài liệu nêu giới hạn đã quan sát, nguồn bên thứ ba, giấy phép đã xác minh/chưa xác minh. |

## 7. Lộ trình 5–7 tuần

Dữ liệu và ground truth làm sớm vì không cần mã tracking.

| Tuần | Việc chính | Mốc nội bộ (trên video dev) |
| --- | --- | --- |
| 1 | Quay/chọn video dev và toàn bộ video test, ghi nguồn + giấy phép; chốt vạch từng video; chốt giao thức ground truth; quyết định về `lap`. | Có video dev; danh sách video test hợp lệ về giấy phép. |
| 2 | `main.py`: detection trên video MP4, xuất video có khung bao. Đếm ground truth lần 1 (video dev, một video test). | Chạy hết video dev, đủ số khung. |
| 3 | Tracking với `bytetrack.yaml`, hiển thị ID, ghi `tracks.csv`. Ground truth lần 1 cho các video test còn lại. | Có video với ID và `tracks.csv`. |
| 4 | `counter.py` + `tests/test_counter.py`; xuất `events.csv`/`counts.json`. Ground truth lần 2 (cách lần 1 ≥ 1 ngày). | Test chạy được; end-to-end ra đủ 3 đầu ra. |
| 5 | `evaluate.py`, đo FPS; **chạy baseline trên tập test.** | Có bảng baseline thật. Đây là mốc tối thiểu có thể nộp. |
| 6 | Thí nghiệm E2 + một trong E1/E4; phân tích lỗi (thiếu/thừa, đổi ID). | Kết quả trên toàn tập test, có nhật ký. |
| 7 | Báo cáo, tái lập, rà soát giấy phép, nghiệm thu AC-1…AC-9; giao diện nếu còn thời gian. | Bảng nghiệm thu điền trung thực. |

**Phương án 5 tuần:** tuần 1 = tuần 1–2 ở trên (dữ liệu, giao thức ground truth, quyết định `lap`, detection trên video); tuần 2 = tracking và bắt đầu logic đếm; tuần 3 = hoàn thiện đếm + unit test, hoàn tất ground truth (đếm hai lần), chạy baseline; tuần 4 = thí nghiệm (chỉ E2 và một trong E1/E4) và phân tích lỗi; tuần 5 = báo cáo, tái lập, nghiệm thu. Cắt: giao diện, fine-tune, E3–E5, đánh giá detection mức khung hình. Không cắt: unit test đếm, ground truth đếm hai lần, baseline, AC-8.

**Kiểm soát tiến độ:** cuối mỗi tuần ghi ngắn việc đã xong/trễ. Điểm quyết định: cuối tuần 2 (detection video chạy?), cuối tuần 3 (tracking chạy?), cuối tuần 5 (có baseline?); trễ quá 3 ngày ở một điểm thì chuyển sang phương án 5 tuần và ghi phần bị cắt vào báo cáo như giới hạn.

**Blocker thường gặp:**
- `lap` không cài/chạy được: chẩn đoán trước trong một môi trường ảo tạm ngoài `.venv`, xin phép chủ dự án trước khi cài; nếu vẫn không được, hỏi chủ dự án về môi trường Python khác; phương án cuối là tracker IoU đơn giản tự viết (ghi rõ là phần tự làm). Không tự thay `lap` bằng `scipy` khi chưa kiểm chứng tương thích. Chưa kiểm tra `tracktrack` có cần `lap` hay không.
- Thiếu video test hợp lệ: dùng video tự quay (ngắn hơn cũng được, giữ tổng ≥ 100 lượt).
- GPU 4 GB không đủ: xem EXPERIMENTS §6.
- Tracker đổi ID nhiều: ghi nhận và đo, không coi là lỗi dự án.
