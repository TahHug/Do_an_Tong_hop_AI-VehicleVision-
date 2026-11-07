# VehicleVision

Hệ thống phát hiện, theo dõi và đếm phương tiện giao thông trong video, sử dụng YOLO26, ByteTrack và OpenCV.

## Giới thiệu

VehicleVision nhận một video giao thông ghi sẵn từ camera cố định và một đoạn vạch đếm do người dùng xác định. Hệ thống phát hiện bốn loại phương tiện (`car`, `motorcycle`, `bus`, `truck`), gán ID cho từng phương tiện qua các khung hình và đếm số lượt phương tiện vượt qua vạch theo từng loại. Kết quả gồm video có chú thích và các tệp thống kê, được đánh giá bằng cách so sánh với số đếm thủ công.

Đây là đồ án tổng hợp học phần Trí tuệ nhân tạo.

## Tính năng

- Phát hiện và phân loại phương tiện bằng YOLO26n pretrained trên COCO.
- Theo dõi nhiều đối tượng bằng ByteTrack, mỗi phương tiện có một ID.
- Đếm lượt vượt một đoạn vạch ảo hữu hạn, mỗi ID tối đa một lần, thống kê theo loại xe.
- Xuất video có chú thích, `tracks.csv`, `events.csv`, `counts.json`.
- Đánh giá sai số đếm (MAE, MAPE, precision/recall mức sự kiện) và tốc độ xử lý.

## Roadmap

- [x] Phát hiện phương tiện trên ảnh
- [ ] Phát hiện phương tiện trên video (đang hoàn thiện)
- [ ] Theo dõi phương tiện (ByteTrack)
- [ ] Đếm lượt vượt vạch
- [ ] Xuất tệp thống kê CSV/JSON
- [ ] Kiểm thử logic đếm
- [ ] Đánh giá độ chính xác và tốc độ

## Công nghệ

| Thành phần | Phiên bản |
| --- | --- |
| Python | 3.14.0 |
| PyTorch / torchvision | 2.13.0+cu132 / 0.28.0+cu132 |
| Ultralytics (YOLO26, ByteTrack) | 8.4.182 |
| OpenCV (`opencv-python`) | 5.0.0.93 |
| NumPy | 2.5.2 |

Môi trường phát triển: Windows 11, NVIDIA RTX 3050 Ti Laptop GPU (4 GB).

## Cấu trúc thư mục

```
VehicleVision/
├── main.py            # chương trình chính: xử lý video
├── test_yolo.py       # kiểm tra nhanh mô hình trên ảnh mẫu
├── requirements.txt
├── docs/
│   ├── DESIGN.md      # kiến trúc, quyết định kỹ thuật, thuật toán đếm
│   └── EVALUATION.md  # dữ liệu, chỉ số, thí nghiệm, kết quả
├── data/
│   ├── ground_truth/  # số lượt đếm thủ công
│   └── videos/        # video đầu vào (không lưu trong repository)
│       ├── dev/       # video phát triển
│       └── eval/      # video đánh giá
└── results/
    ├── demo/          # video kết quả minh họa
    ├── figures/       # ảnh chụp minh họa và biểu đồ số liệu
    ├── baseline/      # kết quả cấu hình chuẩn trên các video đánh giá
    ├── experiments/   # kết quả các thí nghiệm
    └── metrics.csv    # bảng tổng hợp chỉ số
```

Repository không chứa video đầu vào; video được tải từ nguồn công khai (xem mục Dữ liệu) và xử lý trên máy cục bộ. Kết quả của mỗi lần chạy được ghi vào `outputs/` (không lưu trong repository). Các kết quả dùng trong đánh giá và trình bày được lưu trong `results/`.

## Cài đặt

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu132
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

PyTorch bản CUDA được cài trước các thư viện còn lại. Trọng số `yolo26n.pt` được đặt ở thư mục gốc của dự án.

## Sử dụng

```powershell
.\.venv\Scripts\python.exe main.py --video data\videos\dev\day.mp4
```

Thay đường dẫn trong `--video` bằng bất kỳ video nào trong `data/videos/`.

| Tham số | Ý nghĩa | Mặc định |
| --- | --- | --- |
| `--video` | Video đầu vào (MP4) | bắt buộc |
| `--model` | Tệp trọng số YOLO | `yolo26n.pt` |
| `--conf` | Ngưỡng confidence, trong khoảng (0, 1] | `0.35` |
| `--output` | Video đầu ra (MP4) | `outputs\<tên video>_detect.mp4` |

Chương trình ghi video kết quả có khung bao, tên lớp và confidence, giữ nguyên độ phân giải và FPS của video gốc, rồi đọc lại video đầu ra để xác nhận đủ số khung. Thông tin in ra gồm thiết bị xử lý (CUDA/CPU), số khung đọc và ghi, thời gian xử lý và FPS.

`test_yolo.py` chạy mô hình trên một ảnh mẫu để kiểm tra nhanh môi trường.

## Dữ liệu

Video đầu vào là video giao thông ghi sẵn, camera cố định, lấy từ các bộ dữ liệu công khai có giấy phép rõ ràng, và không được lưu trong repository. Video được chia thành hai nhóm: video phát triển (`data/videos/dev/`) dùng để xây dựng và chọn tham số, video đánh giá (`data/videos/eval/`) chỉ dùng để đánh giá. Mọi video đều được xử lý bằng cùng lệnh `main.py`. Tiêu chí chọn dữ liệu, danh sách video đã sử dụng, nguồn và giấy phép của từng bộ dữ liệu: [docs/EVALUATION.md](docs/EVALUATION.md).

## Kết quả

Video kết quả minh họa nằm trong `results/demo/`, ảnh chụp minh họa và biểu đồ trong `results/figures/`, số liệu chi tiết trong `results/baseline/`, `results/experiments/` và `results/metrics.csv`. Phân tích kết quả được trình bày trong [docs/EVALUATION.md](docs/EVALUATION.md).

## Giấy phép và ghi nguồn

| Thành phần | Giấy phép |
| --- | --- |
| [Ultralytics](https://www.ultralytics.com/license) (thư viện và trọng số YOLO26) | AGPL-3.0 |
| [PyTorch](https://pytorch.org) | BSD-3-Clause |
| [OpenCV](https://opencv.org) | Apache 2.0 |
| [NumPy](https://numpy.org) | BSD-3-Clause |
| [lap](https://pypi.org/project/lap/) | BSD-2-Clause |
| [ByteTrack](https://github.com/ifzhang/ByteTrack) (thuật toán; dùng bản cài đặt trong Ultralytics) | MIT |
| [Highway Vehicle Surveillance Dataset](https://data.mendeley.com/datasets/zdgmdg7p25/1) | CC BY 4.0 |

Dự án sử dụng Ultralytics theo giấy phép AGPL-3.0. Mô hình phát hiện và thuật toán theo dõi là thành phần tái sử dụng; thiết kế pipeline, logic đếm, phương pháp và kết quả đánh giá là phần do dự án thực hiện.

## Tài liệu

- [docs/DESIGN.md](docs/DESIGN.md): kiến trúc, quyết định kỹ thuật, thuật toán đếm, định dạng đầu ra.
- [docs/EVALUATION.md](docs/EVALUATION.md): dữ liệu, ground truth, chỉ số đánh giá, thí nghiệm, kết quả.
