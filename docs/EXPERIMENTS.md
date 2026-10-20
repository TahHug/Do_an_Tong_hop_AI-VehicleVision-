# EXPERIMENTS — Kế hoạch đánh giá thực nghiệm

Cập nhật: 2026-10-20.

> **Chưa có thí nghiệm nào được chạy và chưa có số liệu nào.** Tài liệu chỉ định nghĩa dữ liệu, cách đo và quy tắc ghi kết quả. Mọi chỉ số chưa đo ghi là **chưa có kết quả**. Không điền số giả định, không rút kết luận trước khi có dữ liệu thật. Phạm vi và tiêu chí hoàn thành: [PROJECT_PLAN.md](PROJECT_PLAN.md).

## 1. Mục tiêu đánh giá

| Mã | Câu hỏi | Bắt buộc? |
| --- | --- | --- |
| RQ1 | Với YOLO26n pretrained + ByteTrack (`tracker="bytetrack.yaml"`), sai số đếm trên **các video thử nghiệm đã chọn** là bao nhiêu, và quan sát thấy khác nhau thế nào giữa các điều kiện (mật độ, ánh sáng, loại xe)? | Có |
| RQ2 | `yolo26s` so với `yolo26n`: độ chính xác đếm cải thiện bao nhiêu, FPS giảm bao nhiêu trên RTX 3050 Ti 4 GB? | Có (hoặc thay bằng RQ-`imgsz`, §6) |
| RQ3 | Ngưỡng `conf` ảnh hưởng thế nào đến lỗi đếm thiếu/thừa? | Có |
| RQ4 | `bytetrack.yaml`, `botsort.yaml`, `tracktrack.yaml` khác nhau thế nào? | Tùy chọn |
| RQ5 | Điểm tham chiếu (cạnh dưới hay tâm khung bao) ảnh hưởng thế nào? | Tùy chọn |

Dữ liệu nhỏ (vài video, vài trăm lượt xe) nên RQ1 là **đánh giá mô tả**: báo cáo sai số quan sát được trên đúng các video này; không suy ra quan hệ nhân quả, không khái quát cho mọi camera, không dùng kiểm định thống kê. Mọi dự đoán chỉ là giả thuyết cần kiểm chứng.

## 2. Dữ liệu

- **Nguồn ưu tiên: tự quay** từ vị trí cố định, an toàn, hợp pháp (kiểm soát được quyền sử dụng, ground truth đúng với vạch của mình). Tránh công khai video nhận dạng được người/biển số.
- **Dataset công khai** (ví dụ UA-DETRAC, MIO-TCD, AI City Challenge, BDD100K) và **video stock miễn phí**: tên chỉ để tham khảo. **Giấy phép/điều khoản của từng nguồn chưa xác minh**; chỉ dùng sau khi đọc điều khoản và ghi nguồn, giấy phép, ngày tải vào mục §9.
- **Phân chia:** 1 video **dev** (~1 phút) để chỉnh cấu hình và gỡ lỗi; **test** gồm ≥ 3 video (1–3 phút mỗi video, tổng ≥ 100 lượt vượt vạch), nên đa dạng: (V1) thưa, ban ngày; (V2) mật độ cao hoặc nhiều xe máy; (V3) điều kiện khó (ánh sáng yếu, bóng đổ, mưa) nếu có. Mọi tham số (`conf` chọn, `margin_px`, `max_gap_frames`, `Δ` ghép sự kiện) chỉnh trên video dev, **không chỉnh trên video test**.
- Mỗi video ghi: nguồn, độ phân giải, FPS, thời lượng, điều kiện, tọa độ vạch. Tọa độ vạch chốt và lưu **trước** khi đếm ground truth.
- Video, dataset, trọng số không đưa lên Git.

## 3. Ground truth (đếm thủ công)

1. Xem video chậm; mỗi xe vượt vạch ghi một dòng vào `gt_events.csv`: `video_id, frame, time_s, class, direction`.
2. Dùng đúng định nghĩa "vượt vạch" như hệ thống (đoạn di chuyển cắt đoạn vạch; điểm tham chiếu giữa cạnh dưới).
3. Mỗi video đếm **hai lần độc lập**, cách nhau ≥ 1 ngày nếu có thể; xem lại từng khung ở chỗ khác nhau rồi chốt. Ghi số khác biệt giữa hai lần như thước đo độ tin cậy của ground truth.
4. Ghi các trường hợp mơ hồ (che khuất, xe dừng sát vạch) và quyết định.
5. Làm sớm (tuần 1–4), song song với phát triển, vì không cần mã tracking.

## 4. Metrics

### 4.1. Đếm (chính)

Với video `v`: `G_v` là tổng ground truth, `P_v` là tổng hệ thống đếm (chính là `grand_total`); sai số có dấu `e = P - G` (dương = thừa, âm = thiếu). **Đơn vị đánh giá chính: tổng theo video**; đơn vị phụ: `(video, lớp)`, `(video, hướng)`.

| Metric | Công thức | Ghi chú |
| --- | --- | --- |
| MAE | `(1/N) * sum(abs(P_i - G_i))` | Đơn vị: xe/đơn vị. Dùng mọi `N` đơn vị, kể cả `G = 0`. |
| MBE (phụ) | `(1/N) * sum(P_i - G_i)` | Thiên hướng đếm thừa/thiếu. |
| MAPE (chỉ số tương đối) | `(1/N+) * sum(abs(P_i - G_i) / G_i)` trên đơn vị có `G_i > 0` | **Không tính mẫu `G = 0`**; bắt buộc báo `N+` trên `N` và liệt kê đơn vị bị loại. |
| WAPE (tùy chọn, phụ) | `sum(abs(P_i - G_i)) / sum(G_i)` | Khác MAPE: MAPE lấy trung bình tỷ lệ từng đơn vị; WAPE là tổng sai số chia tổng ground truth. Không gộp hai chỉ số. |

Luôn báo `(G, P, e)` của **từng** video trước khi báo giá trị tổng hợp; với vài video chỉ nêu min–max, không báo độ lệch chuẩn hay khoảng tin cậy. Với đơn vị `(video, lớp)` hay có `G` nhỏ hoặc 0: dùng sai số tuyệt đối thay vì phần trăm.

**Mức sự kiện (ghép một-một)** vì số tổng đúng có thể do lỗi thừa và thiếu triệt tiêu nhau. Trong mỗi nhóm (lớp, hướng) của một video: sắp sự kiện hệ thống và GT theo `frame`, ghép tham lam theo thời gian hai con trỏ, chỉ ghép khi `abs(frame_hệ_thống - frame_GT) <= Δ` (mặc định `Δ` = số khung của 1 giây, chọn trên video dev); mỗi sự kiện ở tối đa một cặp. `TP` = số cặp, `FP` = sự kiện hệ thống chưa ghép, `FN` = sự kiện GT chưa ghép; precision/recall/F1 ("n/a" nếu mẫu số 0). Báo cả **class-aware** (đúng lớp và hướng) lẫn **class-agnostic** (chỉ hướng) để thấy phần lỗi do phân loại. Bất biến: `TP + FP = P` và `TP + FN = G` cho mỗi nhóm. Tính chất tối ưu của ghép tham lam trên trục thời gian phải được kiểm tra bằng test khi cài đặt; không cần `scipy`.

### 4.2. Detection và tracking (mức tối thiểu)

- Detection (tùy chọn nếu thiếu thời gian): gán nhãn tay 50–100 khung chọn đều; precision/recall tại IoU 0.5.
- Tracking (bắt buộc): trên đoạn ngắn, quan sát thủ công **số lần đổi ID** và **số đoạn track bị đứt** cho các xe vượt vạch, dựa trên video và `tracks.csv`. Metrics chuẩn như MOTA/IDF1/HOTA cần ground truth tracking đầy đủ, ngoài phạm vi bắt buộc.

## 5. FPS toàn pipeline và cách đo công bằng

**FPS đầu–cuối** = (số khung hình tính sau warm-up) / (thời gian thực, `time.perf_counter()`, từ lúc bắt đầu đọc khung hình đầu tiên được tính đến khi khung hình cuối ghi xong), gồm giải mã, tiền xử lý, suy luận, tracking, đếm, vẽ, ghi video và CSV/JSON. Nếu đo thêm biến thể không ghi video thì gọi tên riêng, không trộn khi so sánh.

- **Đồng bộ CUDA:** GPU chạy bất đồng bộ, nên gọi `torch.cuda.synchronize()` trước mỗi lần đọc đồng hồ ở ranh giới giai đoạn và ở mốc kết thúc.
- **Warm-up:** suy luận vài lần với ảnh giả trước khi đo và bỏ N khung đầu (đề xuất N = 30; ghi N đã dùng).
- **Độ phân giải:** ghi độ phân giải video và `imgsz`; chỉ so FPS giữa các lần chạy cùng video và cùng độ phân giải. Ghi rõ `device` (GPU hay CPU).
- **Số lần lặp:** mỗi (cấu hình, video) chạy toàn bộ ≥ 3 lần; báo **median** và khoảng min–max.
- **Phần cứng:** ghi máy, GPU, driver, cắm điện, power mode **hiện có** của Windows (chỉ ghi nhận, không đổi cài đặt hệ thống), ứng dụng nền, nhiệt độ/xung nếu theo dõi được; chạy từ ổ cục bộ; có thể nghỉ giữa các lần để máy nguội.
- **VRAM đỉnh:** `torch.cuda.max_memory_allocated()` (chỉ phần PyTorch cấp phát), bổ sung `nvidia-smi` để xem tổng; cách đo chưa kiểm chứng.
- Ngưỡng FPS trong PROJECT_PLAN chỉ là mục tiêu tham chiếu, chưa đo.

## 6. Thí nghiệm so sánh (phù hợp GPU 4 GB)

**Baseline:** `yolo26n.pt`; lớp COCO 2, 3, 5, 7; `conf=0.35`; `tracker="bytetrack.yaml"` truyền tường minh (tham số lấy từ `bytetrack.yaml` đi kèm Ultralytics 8.4.182, không ghi đè nếu chưa có lý do); `imgsz` mặc định (640); điểm tham chiếu giữa cạnh dưới; GPU CUDA. Chạy baseline trên toàn tập test trước khi đổi bất kỳ biến nào.

Mỗi thí nghiệm đổi **một** biến:

| Mã | Biến | Mức | Bắt buộc? |
| --- | --- | --- | --- |
| E1 | Mô hình | `yolo26n` (baseline), `yolo26s` | Có (hoặc thay bằng E4) |
| E2 | `conf` | 0.25, 0.35 (baseline), 0.50 | Có |
| E3 | Tracker | `bytetrack.yaml`, `botsort.yaml`, `tracktrack.yaml` (mặc định thư viện) | Tùy chọn |
| E4 | `imgsz` | 640 (baseline), 480 và/hoặc 960 nếu VRAM cho phép | Tùy chọn; thay thế E1 |
| E5 | Điểm tham chiếu | cạnh dưới (baseline), tâm khung bao | Tùy chọn; dự phòng cuối |

**Bảo đảm tối thiểu hai thí nghiệm kiểm chứng được:** E2 (luôn khả thi) cộng một trong E1/E4. Việc `yolo26s` vừa 4 GB, FP16, BoT-SORT, TrackTrack chạy được chưa được thử; `yolo26s.pt` chưa có trong máy. Thứ tự dự phòng: (1) E1 như trên, batch = 1; (2) nếu `yolo26s` hết VRAM hoặc quá chậm, bật `half=True` cho **cả hai** mô hình để biến duy nhất vẫn là kích thước mô hình, và ghi nhận `half` có chạy/giữ chất lượng hay không; (3) nếu vẫn không được hoặc không tải được `yolo26s.pt`, thay E1 bằng E4; (4) nếu cả hai không khả thi, dùng E5 và ghi lý do. Không để số thí nghiệm xuống dưới hai mà không ghi lại.

## 7. Quy tắc lưu kết quả

Mỗi lần chạy lưu ở `outputs/<mã thí nghiệm>_<video>/` (không lên Git) gồm video, `tracks.csv`, `events.csv`, `counts.json`, và một dòng trong nhật ký thí nghiệm với:

`mã thí nghiệm · ngày · video · lệnh chạy · git commit · phiên bản thư viện (Python, PyTorch, Ultralytics, OpenCV, NumPy) · tệp trọng số · tracker: tên tệp, phiên bản Ultralytics, nội dung cấu hình tracker thực tế và tham số bị ghi đè · vạch (a, b, margin_px, max_gap_frames) · phần cứng và chế độ nguồn · ghi chú`

Không ghi chung chung "tracker mặc định". Bảng kết quả (mẫu, để trống đến khi có dữ liệu thật):

| Thí nghiệm | Video | Cấu hình (gồm tracker) | G | P | e = P - G | Sai số % (nếu G > 0) | FPS (median) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (chưa có kết quả) | | | | | | | |

## 8. Giới hạn và rủi ro ảnh hưởng đến kết luận

- Tập test nhỏ: kết luận chỉ mang tính chỉ báo.
- Chỉnh tham số trên chính video test làm sai lệch kết quả: tránh bằng video dev riêng.
- Ground truth do một người tạo, có thể sai/thiên lệch: giảm bằng đếm hai lần.
- Mô hình pretrained COCO không được huấn luyện cho góc nhìn/loại xe cụ thể (đặc biệt xe máy dày đặc, che khuất, ban đêm): không khái quát cho mọi camera.
- Đếm theo ID: một xe có hai ID có thể đếm hai lần, hai xe chung ID đếm thiếu; đổi ID ngay lúc vượt vạch không có quy tắc cục bộ nào sửa được.
- FPS phụ thuộc phần cứng, nhiệt độ, power mode của laptop; chỉ so sánh trong cùng điều kiện.
- Môi trường rất mới (Python 3.14, OpenCV 5.0, NumPy 2.5, PyTorch 2.13): chưa kiểm chứng đọc/ghi video và tracking; `lap` chưa cài và ByteTrack của Ultralytics cần `lap` (tracker `tracktrack` chưa kiểm tra).

## 9. Nguồn học thuật và dữ liệu đã đọc (2026-10-19)

**Đã đối chiếu nguồn:**
- Ultralytics, YOLO26: https://docs.ultralytics.com/models/yolo26/ (5 cỡ n/s/m/l/x, 80 lớp COCO; ngày phát hành trên trang không nhất quán nên không trích).
- Ultralytics, `default.yaml` tại tag v8.4.182: `tracker: tracktrack.yaml` (https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.182/ultralytics/cfg/default.yaml; đã đối chiếu với bản trong `.venv`).
- **ByteTrack:** Y. Zhang, P. Sun, Y. Jiang, D. Yu, F. Weng, Z. Yuan, P. Luo, W. Liu, X. Wang, "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", arXiv:2110.06864, https://arxiv.org/abs/2110.06864 (gửi 13/10/2021, v3 sửa 7/4/2022). Trang arXiv không ghi hội nghị; "ECCV 2022" lấy từ README của tác giả (https://github.com/ifzhang/ByteTrack), **chưa đối chiếu** ECVA/Springer.
- **Microsoft COCO:** T.-Y. Lin, M. Maire, S. Belongie, L. Bourdev, R. Girshick, J. Hays, P. Perona, D. Ramanan, C. L. Zitnick, P. Dollár, "Microsoft COCO: Common Objects in Context", arXiv:1405.0312, https://arxiv.org/abs/1405.0312 (gửi 1/5/2014, v3 sửa 21/2/2015). Trang arXiv không ghi hội nghị; việc đăng ở ECCV 2014 **chưa xác minh** (trang Springer chuyển hướng đăng nhập).
- PyPI, `lap`: https://pypi.org/project/lap/ (0.5.13, BSD-2-Clause, có wheel `cp314 win_amd64`).

**Chưa xác minh:** giấy phép/điều khoản của COCO, torchvision và mọi dataset/video ứng viên ở §2; bộ dữ liệu cụ thể sẽ dùng chưa chọn. Giấy phép các thành phần phần mềm: xem README.

**Dữ liệu thử nghiệm đã có:** chưa có video test hay ground truth nào.
