# VehicleVision

Chương trình **phát hiện, theo dõi và đếm phương tiện giao thông trong video** bằng YOLO (Ultralytics) và OpenCV. Đồ án tổng hợp cá nhân (môn Computer Vision / AI, 5–7 tuần).

## Mục tiêu

Với một video giao thông ghi sẵn (MP4) quay từ camera cố định, chương trình cần:

1. Phát hiện và phân loại xe: ô tô (`car`), xe máy (`motorcycle`), xe buýt (`bus`), xe tải (`truck`).
2. Gán ID cho từng xe qua các khung hình (tracking).
3. Đếm số xe vượt qua một đoạn vạch ảo, theo loại xe (lượt vượt từ cả hai phía cùng vào một tổng, không phân nhóm theo hướng), hạn chế đếm trùng theo ID.
4. Xuất video đã chú thích cùng CSV/JSON thống kê.
5. Đo độ chính xác đếm (so với đếm thủ công) và tốc độ xử lý.

## Công nghệ

Python 3.14, PyTorch (CUDA), Ultralytics YOLO26 (mô hình `yolo26n` pretrained), ByteTrack (có sẵn trong Ultralytics), OpenCV, NumPy.

## Trạng thái (cập nhật 2026-10-20)

| Chức năng | Trạng thái |
| --- | --- |
| Môi trường Python, PyTorch CUDA, Ultralytics, OpenCV | Đã cài và dùng được |
| Phát hiện xe trên **ảnh** (`test_yolo.py`) | Đã chạy thành công (theo chủ dự án; ảnh kết quả trong `runs/`) |
| Phát hiện xe trên **video** | **Chưa làm** |
| Tracking | **Chưa làm** |
| Đếm qua vạch ảo | **Chưa làm** |
| Xuất video, CSV/JSON | **Chưa làm** |
| Đánh giá sai số và FPS | **Chưa làm** (chưa có số liệu nào) |

## Thiết lập môi trường (Windows 11, PowerShell)

Môi trường đã kiểm tra: Python 3.14.0, PyTorch 2.13.0+cu132, torchvision 0.28.0+cu132, Ultralytics 8.4.182, opencv-python 5.0.0.93, NumPy 2.5.2, GPU NVIDIA RTX 3050 Ti Laptop (4 GB). Dựng lại trên máy mới (các bước này chưa được chạy lại trên máy sạch):

```powershell
py -3.14 -m venv .venv
# 1) Cài PyTorch bản CUDA TRƯỚC (nếu cài ultralytics trước, pip có thể kéo bản CPU)
.\.venv\Scripts\python.exe -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu132
# 2) Cài các thư viện còn lại (đã ghim phiên bản)
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Nếu PowerShell chặn `Activate.ps1`, không cần đổi chính sách hệ thống: luôn gọi `.\.venv\Scripts\python.exe` trực tiếp.

Kiểm tra (chỉ đọc, không sửa môi trường):

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Lần chạy `pip check` ngày 2026-10-19 trên `.venv` hiện tại: `No broken requirements found.` (chỉ cho biết ràng buộc phiên bản không xung đột, không chứng minh chức năng chạy đúng). Danh sách `pip freeze` đầy đủ (40 gói) của ngày đó nằm trong bản sao lưu ZIP cục bộ của chủ dự án, ngoài Git; có thể tạo lại bằng `.\.venv\Scripts\python.exe -m pip freeze`.

### Lưu ý về môi trường

- **Tracker mặc định của Ultralytics 8.4.182 là `tracktrack.yaml`, không phải ByteTrack.** Dự án chủ động dùng `tracker="bytetrack.yaml"`. Cách xác nhận bằng thao tác chỉ đọc:
  `Select-String -Path .\.venv\Lib\site-packages\ultralytics\cfg\default.yaml -Pattern '^tracker:'`
- **`lap` chưa cài.** `ultralytics/trackers/utils/matching.py` import `lap`, nếu thiếu Ultralytics sẽ cố tự `pip install` khi chạy. Vì vậy **chưa chạy tracking** cho đến khi có quyết định về `lap` (PyPI có `lap 0.5.13` với wheel cho CPython 3.14 Windows x64, nhưng chưa thử ở đây). `pytest` và `scipy` cũng chưa cài.
- Kết quả của `test_yolo.py` đang nằm ở `runs/detect/runs/detect/first_test/` (thư mục lồng nhau; quan sát được, chưa tìm nguyên nhân).

## Chạy `test_yolo.py`

```powershell
.\.venv\Scripts\python.exe test_yolo.py
```

Script tải ảnh mẫu từ Internet, chạy YOLO26n với các lớp COCO 2, 3, 5, 7 (`car`, `motorcycle`, `bus`, `truck`), `conf=0.35`, và lưu ảnh có khung bao vào `runs/`. Lần đầu có thể tải trọng số `yolo26n.pt`.

## Chạy chương trình video

**Chưa có.** Chương trình xử lý video (`main.py`) chưa được viết nên chưa có lệnh để chạy. Khi hoàn thành, lệnh sẽ được ghi vào đây.

## Dữ liệu vào, kết quả ra (thiết kế, chưa triển khai)

- **Vào:** một video giao thông `.mp4` ghi sẵn, từ camera cố định, lấy từ bộ dữ liệu công khai hoặc nguồn có giấy phép kiểm chứng được (không hỗ trợ webcam, camera thời gian thực, RTSP hay livestream); tọa độ hai đầu của vạch đếm; tên mô hình; ngưỡng `conf`; tên tracker. Tiêu chí chọn video, cách tách video dev/evaluation và cách ghi nguồn: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) §2.
- **Ra** (trong `outputs/`): video đã chú thích (khung bao, ID, vạch, bộ đếm), `tracks.csv` (mọi track theo khung hình), `events.csv` (mỗi lượt vượt vạch), `counts.json` (tổng hợp và thông tin lần chạy).

Video, dữ liệu, trọng số và kết quả sinh ra **không** đưa lên Git (`.gitignore`).

## Nguồn và giấy phép bên thứ ba (đã kiểm tra ngày 2026-10-19)

| Thành phần | Giấy phép | Nguồn thông tin |
| --- | --- | --- |
| Ultralytics 8.4.182 (thư viện) | AGPL-3.0 | [ultralytics.com/license](https://www.ultralytics.com/license); siêu dữ liệu gói |
| Trọng số YOLO26 (`yolo26n.pt`) | Ultralytics nêu: mô hình do Ultralytics huấn luyện thuộc AGPL-3.0 theo mặc định; có giấy phép Enterprise trả phí | [ultralytics.com/license](https://www.ultralytics.com/license), [YOLO26](https://docs.ultralytics.com/models/yolo26/) |
| PyTorch 2.13.0 | Siêu dữ liệu gói ghi tổ hợp giấy phép cho phép (Apache-2.0, BSD, MIT...); chưa đối chiếu trang chính thức | [pytorch.org](https://pytorch.org) |
| OpenCV (`opencv-python` 5.0.0.93) | Siêu dữ liệu gói ghi Apache 2.0; chưa rà soát `LICENSE-3RD-PARTY.txt` | [opencv.org](https://opencv.org) |
| NumPy 2.5.2 | BSD-3-Clause và các giấy phép cho phép khác (siêu dữ liệu gói) | [numpy.org](https://numpy.org) |
| `lap` (chưa cài, dự kiến 0.5.13) | BSD-2-Clause | [pypi.org/project/lap](https://pypi.org/project/lap/) |
| ByteTrack (mã gốc của tác giả) | MIT (repository gốc); dự án dùng bản cài đặt trong Ultralytics (AGPL-3.0), không dùng mã gốc | [github.com/ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack), [arXiv:2110.06864](https://arxiv.org/abs/2110.06864) |
| Highway Vehicle Surveillance Dataset (video dev, ví dụ `day.mp4`) — H. Sadoghi Yazdi, Mendeley Data, v1, 2025, DOI 10.17632/zdgmdg7p25.1 | CC BY 4.0 (phải ghi nguồn) | [data.mendeley.com/datasets/zdgmdg7p25/1](https://data.mendeley.com/datasets/zdgmdg7p25/1) (đã đọc 2026-10-24) |
| COCO (dữ liệu huấn luyện của mô hình gốc) | **Chưa xác minh** | [arXiv:1405.0312](https://arxiv.org/abs/1405.0312), [cocodataset.org](https://cocodataset.org/) |
| torchvision, các thư viện phụ thuộc gián tiếp, video/dataset thử nghiệm | **Chưa xác minh** | — |

**Giấy phép của VehicleVision: chưa chọn, chưa có tệp `LICENSE`.** Không tự gán MIT.

- *Phát biểu của Ultralytics (nguồn chính thức, không phải tư vấn pháp lý):* AGPL-3.0 dành cho nghiên cứu học thuật, bài tập, dự án cá nhân và ứng dụng mã nguồn mở công bố đầy đủ; tuân thủ AGPL nghĩa là công bố toàn bộ mã nguồn tương ứng của tác phẩm phái sinh; dùng "chỉ nội bộ hoặc R&D" vẫn cần mở mã hoặc giấy phép Enterprise. Văn bản AGPL xác nhận quyền chạy chương trình chưa sửa đổi, và phân phối bản sửa đổi thì toàn bộ tác phẩm phải theo AGPL-3.0.
- *Diễn giải của người soạn (chưa xác nhận):* nếu VehicleVision `import ultralytics` và được công khai/phân phối, toàn bộ dự án có thể phải theo AGPL-3.0; việc nộp bài cho giảng viên có tính là "phân phối" hay không **chưa được xác nhận**. Cần hỏi giảng viên/khoa trước khi công khai repository.
- Phần **sinh viên tự làm** (logic đếm, đánh giá, thí nghiệm, báo cáo) tách biệt với phần **tái sử dụng** (mô hình, tracker, thư viện).

## Tài liệu

- [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md): phạm vi, pipeline, quyết định kỹ thuật, tiêu chí hoàn thành, lộ trình.
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md): dữ liệu, ground truth, metrics, đo FPS, thí nghiệm, quy tắc lưu kết quả.
