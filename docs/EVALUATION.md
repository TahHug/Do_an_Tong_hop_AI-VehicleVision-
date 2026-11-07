# Đánh giá hệ thống VehicleVision

Tài liệu mô tả dữ liệu, ground truth, chỉ số đánh giá, phương pháp đo tốc độ, các thí nghiệm và kết quả. Thiết kế hệ thống: [DESIGN.md](DESIGN.md).

## 1. Mục tiêu đánh giá

| Mã | Câu hỏi |
| --- | --- |
| RQ1 | Với YOLO26n pretrained và ByteTrack, sai số đếm trên các video đánh giá là bao nhiêu, và khác nhau thế nào giữa các điều kiện (mật độ, ánh sáng, loại xe)? |
| RQ2 | Kích thước mô hình (`yolo26n` so với `yolo26s`) hoặc kích thước ảnh đầu vào ảnh hưởng thế nào đến độ chính xác đếm và FPS trên GPU 4 GB? |
| RQ3 | Ngưỡng `conf` ảnh hưởng thế nào đến lỗi đếm thiếu và đếm thừa? |

Tập đánh giá gồm một số ít video, nên kết quả mang tính mô tả: sai số được báo cáo trên chính các video này, không khái quát cho mọi camera và không dùng kiểm định thống kê.

## 2. Dữ liệu

**Tiêu chí chọn video:**
- Video giao thông ghi sẵn, định dạng MP4, từ bộ dữ liệu công khai có giấy phép rõ ràng.
- Camera cố định, góc nhìn ổn định, không chuyển cảnh; đường giao thông thực tế, đủ sáng.
- Có các lớp `car`, `motorcycle`, `bus`, `truck`; xác định được một đoạn vạch đếm phù hợp.
- Video gốc, chưa bị vẽ khung bao, ID hay kết quả nhận diện.
- Điều kiện từ đơn giản đến vừa phải; điều kiện cực đoan (camera rung mạnh, trời tối, mưa lớn, ùn tắc phức tạp) nằm ngoài phạm vi đánh giá.

**Tổ chức dữ liệu:**
- **Video phát triển:** video ngắn, hình ảnh rõ; dùng để xây dựng, kiểm tra từng chức năng và chọn tham số (`conf`, `margin_px`, `max_gap_frames`, `Δ`).
- **Video đánh giá:** ít nhất 3 video độc lập, mỗi video 1–3 phút, tổng ít nhất 100 lượt vượt vạch, nhiều phương tiện và đa dạng loại xe hơn video phát triển. Chỉ dùng để đánh giá, không dùng để chọn tham số.
- Một video, kể cả các đoạn cắt từ cùng một video gốc, chỉ thuộc một nhóm.

**Dữ liệu sử dụng:**

| Video | Nhóm | Nguồn | Thông số |
| --- | --- | --- | --- |
| `dev/day.mp4` | Phát triển | Highway Vehicle Surveillance Dataset (điều kiện nắng) | 3840×2160, 30,17 FPS, 149 khung (4,9 s), camera cố định trên cao tốc |
| `dev/rain.mp4` | Phát triển | Highway Vehicle Surveillance Dataset (điều kiện mưa) | 704×432, 30,13 FPS, 194 khung (6,4 s), camera cố định trên cao tốc, mặt đường ướt |

**Nguồn dữ liệu:**

1. **Highway Vehicle Surveillance Dataset:** H. Sadoghi Yazdi, Ferdowsi University of Mashhad; Mendeley Data, phiên bản 1, 2025; DOI [10.17632/zdgmdg7p25.1](https://data.mendeley.com/datasets/zdgmdg7p25/1); giấy phép CC BY 4.0. Bộ dữ liệu gồm video ở điều kiện nắng, đêm, mưa cùng kết quả xử lý của tác giả (khung bao, ID); kết quả xử lý này không được dùng làm ground truth.

**Lưu trữ:** video đầu vào không có trong repository và được xử lý trên máy cục bộ: video phát triển nằm trong `data/videos/dev/`, video đánh giá trong `data/videos/eval/`. Bảng "Dữ liệu sử dụng" ghi cho mỗi video: nhóm, nguồn, độ phân giải, FPS, thời lượng và điều kiện ghi hình; nếu video là đoạn cắt từ video gốc thì ghi thêm vị trí đoạn cắt. Tọa độ vạch đếm của mỗi lần chạy được lưu trong `counts.json` ([DESIGN.md](DESIGN.md)).

## 3. Ground truth

1. Mỗi lượt vượt vạch được ghi một dòng vào `gt_events.csv`: `video_id, frame, time_s, class`.
2. Định nghĩa vượt vạch giống hệ thống: đoạn di chuyển của điểm tham chiếu (trung điểm cạnh dưới khung bao) cắt đoạn vạch. Tọa độ vạch được cố định trước khi đếm.
3. Mỗi video được đếm thủ công hai lần độc lập, cách nhau ít nhất một ngày; các điểm khác nhau được xem lại từng khung hình rồi chốt. Số khác biệt giữa hai lần đếm được báo cáo như thước đo độ tin cậy của ground truth.
4. Các trường hợp mơ hồ (che khuất, phương tiện dừng sát vạch) được ghi lại cùng cách xử lý.

## 4. Chỉ số đánh giá

### 4.1. Sai số đếm

Với video `v`: `G_v` là số lượt theo ground truth, `P_v` là số lượt hệ thống đếm (`grand_total`), `e = P − G` (dương: đếm thừa, âm: đếm thiếu). Đơn vị đánh giá chính là tổng theo video; đơn vị phụ là cặp `(video, lớp)`.

| Chỉ số | Công thức | Ý nghĩa |
| --- | --- | --- |
| MAE | `(1/N) · Σ abs(P_i − G_i)` | Sai số tuyệt đối trung bình (đơn vị: xe), tính trên mọi đơn vị. |
| MBE | `(1/N) · Σ (P_i − G_i)` | Xu hướng đếm thừa hay đếm thiếu. |
| MAPE | `(1/N⁺) · Σ abs(P_i − G_i) / G_i`, trên các đơn vị có `G_i > 0` | Sai số tương đối trung bình; đơn vị có `G = 0` không được tính, số đơn vị hợp lệ `N⁺` được báo cáo kèm. |
| WAPE | `Σ abs(P_i − G_i) / Σ G_i` | Tổng sai số so với tổng ground truth (chỉ số phụ). |

Kết quả `(G, P, e)` của từng video được báo cáo trước giá trị tổng hợp; với số video ít, chỉ nêu giá trị nhỏ nhất và lớn nhất. Với đơn vị `(video, lớp)` có `G` nhỏ hoặc bằng 0, dùng sai số tuyệt đối.

### 4.2. Đánh giá mức sự kiện

Tổng số đếm có thể đúng do lỗi thừa và lỗi thiếu bù trừ nhau, nên từng sự kiện cũng được đối chiếu. Trong mỗi video, sự kiện của hệ thống và của ground truth được sắp theo thời gian và ghép một-một, chỉ ghép khi `abs(frame_hệ_thống − frame_GT) ≤ Δ` (mặc định `Δ` bằng số khung trong 1 giây). Số cặp ghép được là `TP`, sự kiện hệ thống chưa ghép là `FP`, sự kiện ground truth chưa ghép là `FN`; từ đó tính precision, recall và F1. Kết quả được báo cáo ở hai mức: **theo lớp** (chỉ ghép sự kiện cùng lớp) và **không theo lớp**, để tách lỗi phân loại khỏi lỗi đếm. Kiểm tra nhất quán: `TP + FP = P` và `TP + FN = G`.

### 4.3. Phát hiện và theo dõi

- Theo dõi: số lần đổi ID và số lần track bị đứt của các phương tiện vượt vạch, đếm thủ công trên video kết quả và `tracks.csv`.
- Phát hiện (bổ sung): precision và recall tại IoU 0,5 trên 50–100 khung hình được gán nhãn thủ công.

## 5. Đo tốc độ xử lý

**FPS toàn pipeline** = số khung hình được tính / thời gian thực (đo bằng `time.perf_counter()`) từ khi đọc khung đầu tiên được tính đến khi ghi xong khung cuối, bao gồm giải mã, suy luận, theo dõi, đếm, vẽ và ghi kết quả.

- Gọi `torch.cuda.synchronize()` trước mỗi lần đọc thời gian, vì GPU xử lý bất đồng bộ.
- Chạy khởi động (warm-up) trước khi đo và bỏ 30 khung đầu.
- Ghi độ phân giải video và `imgsz`; chỉ so sánh FPS giữa các lần chạy cùng video và cùng độ phân giải.
- Mỗi cấu hình chạy ít nhất 3 lần trên mỗi video; báo cáo median và khoảng min–max.
- Ghi điều kiện phần cứng: GPU, driver, nguồn điện, chế độ nguồn của hệ điều hành.
- Bộ nhớ GPU: `torch.cuda.max_memory_allocated()` và `nvidia-smi`.

## 6. Thiết lập thí nghiệm

**Cấu hình baseline:** `yolo26n.pt`; lớp COCO 2, 3, 5, 7; `conf=0.35`; `tracker="bytetrack.yaml"` với tham số mặc định của Ultralytics 8.4.182; `imgsz=640`; điểm tham chiếu là trung điểm cạnh dưới khung bao; GPU CUDA.

**Mục tiêu tham chiếu** (trên video đánh giá):
- MAPE ≤ 15 % trên video điều kiện tốt.
- FPS toàn pipeline ≥ 10 với `yolo26n`, đầu vào ≤ 1280×720, trên RTX 3050 Ti.

**Thí nghiệm độ nhạy cấu hình.** Mỗi thí nghiệm thay đổi một biến so với baseline:

| Mã | Biến | Các mức |
| --- | --- | --- |
| E1 | Mô hình | `yolo26n` (baseline), `yolo26s` |
| E2 | `conf` | 0,25; 0,35 (baseline); 0,50 |
| E3 | Tracker | `bytetrack.yaml` (baseline), `botsort.yaml`, `tracktrack.yaml` |
| E4 | `imgsz` | 640 (baseline), 480, 960 |
| E5 | Điểm tham chiếu | Trung điểm cạnh dưới (baseline), tâm khung bao |

Các thí nghiệm chính là E2 và một trong E1/E4; E3 và E5 là bổ sung. Với GPU 4 GB, E1 chạy với batch = 1; nếu `yolo26s` không đủ bộ nhớ, cả hai mô hình cùng chạy ở chế độ `half=True` để kích thước mô hình vẫn là biến duy nhất; nếu vẫn không khả thi, E4 được dùng thay cho E1.

**Lưu kết quả:** mỗi lần chạy ghi vào `outputs/`. Kết quả dùng trong đánh giá được lưu trong `results/`: `baseline/<video>/` và `experiments/<mã thí nghiệm>/<video>/` chứa `counts.json`, `events.csv`; `metrics.csv` tổng hợp chỉ số của mọi lần chạy; `demo/` chứa video kết quả minh họa; `figures/` chứa ảnh chụp minh họa và biểu đồ số liệu.

**Ghi nhận mỗi lần chạy:** mã thí nghiệm, ngày, video, lệnh chạy, commit, phiên bản thư viện, tệp trọng số, tracker và tham số, tọa độ vạch và tham số đếm, điều kiện phần cứng.

## 7. Kết quả

| Thí nghiệm | Video | Cấu hình | G | P | e = P − G | Sai số % | FPS (median) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| — | — | — | — | — | — | — | — |

## 8. Giới hạn

- Tập đánh giá nhỏ nên kết luận mang tính chỉ báo.
- Góc camera, độ phân giải, mức nén và điều kiện ghi hình của video phụ thuộc vào nguồn dữ liệu; kết quả chỉ phản ánh các video đã chọn.
- Ground truth do một người tạo; sai lệch được giảm bằng đếm hai lần.
- Mô hình pretrained trên COCO không được huấn luyện riêng cho camera giao thông; chất lượng có thể giảm khi xe máy dày đặc, bị che khuất hoặc thiếu sáng.
- Đếm theo ID chịu ảnh hưởng của lỗi tracking: phương tiện bị gán hai ID có thể bị đếm hai lần, hai phương tiện chung ID bị đếm thiếu.
- FPS phụ thuộc phần cứng, nhiệt độ và chế độ nguồn của máy.
