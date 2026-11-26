# Thiết kế hệ thống VehicleVision

## 1. Bài toán

Cho một video giao thông ghi sẵn từ camera cố định và một đoạn vạch đếm, xác định số lượt phương tiện vượt qua vạch theo từng loại xe.

- **Đầu vào:** video MP4; tọa độ hai đầu `a`, `b` của đoạn vạch đếm; mô hình, ngưỡng confidence, tracker.
- **Đầu ra:** video có chú thích (khung bao, ID, vạch đếm, bộ đếm theo loại xe), `tracks.csv`, `events.csv`, `counts.json`.
- **Lớp phương tiện:** `car`, `motorcycle`, `bus`, `truck` (lớp COCO 2, 3, 5, 7).

**Phạm vi.** Hệ thống xử lý một camera cố định với một đoạn vạch đếm, trên video ghi sẵn. Lượt vượt từ hai phía của vạch được cộng chung một tổng; hệ thống không phân loại hướng di chuyển. Không thuộc phạm vi: nhận dạng biển số, đo tốc độ, phát hiện vi phạm, nhiều vạch đếm, camera chuyển động, nguồn video trực tiếp (webcam, RTSP, livestream), huấn luyện mô hình.

## 2. Kiến trúc

```text
video MP4 -> Detection + Tracking (YOLO26n + ByteTrack) -> Counting (vạch ảo) -> Output
                                                                          |
              Evaluation (chạy riêng): events/counts  +  ground truth  ---+
```

Với mỗi khung hình: ảnh → danh sách track (ID, khung bao, lớp, confidence) → các sự kiện vượt vạch → vẽ lên video, ghi `tracks.csv` và `events.csv`. Kết thúc video, ghi `counts.json`.

| Thành phần | Vai trò |
| --- | --- |
| `main.py` | Đọc video, gọi mô hình phát hiện và theo dõi, gọi bộ đếm, vẽ chú thích, ghi video và tệp thống kê. Tham số truyền qua dòng lệnh. |
| `counter.py` | Logic đếm vượt vạch, viết bằng Python/NumPy, độc lập với YOLO và OpenCV. |
| `evaluate.py` | So sánh kết quả đếm với ground truth, tính các chỉ số trong [EVALUATION.md](EVALUATION.md). |
| `tests/test_counter.py` | Unit test cho logic đếm (pytest). |
| `test_yolo.py` | Kiểm tra nhanh mô hình trên ảnh mẫu. |

## 3. Quyết định kỹ thuật

**Mô hình phát hiện: YOLO26n pretrained trên COCO, `conf=0.35`.**
COCO đã có đủ bốn lớp phương tiện cần dùng nên không cần huấn luyện lại. YOLO26n là cỡ nhỏ và nhanh nhất của họ YOLO26, phù hợp GPU 4 GB. Ngưỡng `conf=0.35` là mức confidence để một phương tiện được cấp ID, là giá trị khởi đầu và được phân tích thêm trong thí nghiệm E2.

**Tracker: ByteTrack (`bytetrack.yaml`).**
ByteTrack ghép cả các khung bao có độ tin cậy thấp vào các track đang có, giúp giữ ID khi phương tiện bị che khuất một phần. Thuật toán có sẵn trong Ultralytics (cần thư viện `lap`). Tracker được chỉ định tường minh vì giá trị mặc định có thể khác giữa các phiên bản (Ultralytics 8.4.182 mặc định dùng `tracktrack.yaml`).

Ở mỗi khung hình, ByteTrack dự đoán vị trí mới của các track bằng bộ lọc Kalman, rồi ghép track với khung bao theo hai bước:

1. Khung bao có confidence từ `track_high_thresh` trở lên được ghép trước, theo độ chồng lấn IoU (có kết hợp confidence, `fuse_score`).
2. Các track đang theo dõi mà chưa ghép được thử ghép tiếp, chỉ theo IoU, với khung bao có confidence từ trên `track_low_thresh` đến dưới `track_high_thresh`.

Track mới chỉ được tạo từ khung bao chưa ghép có confidence từ `new_track_thresh` trở lên. Track không ghép được quá `track_buffer` khung bị loại bỏ.

| Tham số | Giá trị | Nguồn |
| --- | --- | --- |
| `track_high_thresh`, `new_track_thresh` | bằng `--conf` (0,35) | đặt theo lần chạy |
| `track_low_thresh` | 0,1 | `bytetrack.yaml` |
| `track_buffer` | 30 | `bytetrack.yaml` |
| `match_thresh` | 0,8 | `bytetrack.yaml` |
| `fuse_score` | bật | `bytetrack.yaml` |

Bộ phát hiện lọc khung bao ở mức `track_low_thresh`, còn `--conf` được ghi vào `track_high_thresh` và `new_track_thresh` trong tệp `tracker.yaml` của lần chạy. *Lý do:* nếu bộ phát hiện lọc ngay ở 0,35 thì tracker không nhận được khung bao điểm thấp nào và bước ghép thứ hai không bao giờ chạy. Vì vậy một track đã có ID có thể có `conf` dưới 0,35 ở một số khung hình.

**Phát hiện và theo dõi trong một lời gọi `model.track(...)`.**
Giảm lượng mã và đơn giản hóa kiểm tra. Đánh đổi: không đo tách riêng được thời gian của hai bước.

**Điểm tham chiếu: trung điểm cạnh dưới khung bao `((x1+x2)/2, y2)`.**
Điểm này gần vị trí bánh xe tiếp xúc mặt đường, nên thời điểm vượt vạch ít phụ thuộc vào chiều cao xe và góc nhìn hơn so với tâm khung bao.

**Vạch đếm là đoạn thẳng hữu hạn `a`–`b`.**
Cho phép chỉ đếm trong khu vực đã chọn, ví dụ một chiều đường, và không đếm phương tiện đi ngoài hai đầu vạch. Vạch suy biến (`a == b`) là lỗi cấu hình.

**Đếm theo ID, mỗi ID tối đa một lượt.**
Một phương tiện xuất hiện trong nhiều khung hình liên tiếp, nên lượt đếm được gắn với ID để tránh đếm lặp. Một track ID không đồng nghĩa với một phương tiện thật: phương tiện bị gán hai ID có thể bị đếm hai lần, hai phương tiện chung một ID chỉ được đếm một lần. Sai lệch này được đo ở mức sự kiện trong đánh giá.

## 4. Thuật toán đếm

**Xác định phía của một điểm.** Với điểm tham chiếu `p`, khoảng cách có dấu tới đường thẳng qua `a`, `b` là

`d(p) = cross(b − a, p − a) / |b − a|`

Dấu của tích có hướng cho biết `p` nằm ở phía nào của vạch. Một vùng đệm `margin_px` quanh vạch giúp bỏ qua dao động nhỏ của khung bao:

`side(p) = +1 nếu d(p) > margin_px;  −1 nếu d(p) < −margin_px;  0 nếu nằm trong vùng đệm`

**Trạng thái theo ID:** `last_side` (phía khác 0 gần nhất), `last_point` (vị trí tương ứng), trạng thái đã đếm và lịch sử lớp.

**Quy tắc cập nhật cho mỗi track ở mỗi khung hình:**

1. `side == 0` (chạm vạch hoặc trong vùng đệm): giữ nguyên trạng thái.
2. ID chưa có `last_side`: ghi nhận vị trí, chưa đếm.
3. `side` ngược dấu `last_side`: nếu đoạn `last_point → p` cắt đoạn vạch hữu hạn và ID chưa được đếm thì tạo một sự kiện. Nếu phương tiện đi vòng qua đầu mút vạch thì không đếm. Trong cả hai trường hợp, trạng thái được cập nhật.
4. ID vắng mặt không quá `max_gap_frames` khung rồi xuất hiện lại: vẫn xét đoạn nối hai vị trí quan sát. Vắng lâu hơn: trạng thái được đặt lại.
5. Mỗi ID một sự kiện; các lần vượt lại sau đó chỉ được ghi nhận (`recrossings`), không cộng vào tổng.
6. Lớp của sự kiện là lớp xuất hiện nhiều nhất trong lịch sử của track đến lúc vượt vạch; nếu bằng nhau, lấy lớp gần nhất.

**Ảnh hưởng của lỗi tracking:**

| Tình huống | Ảnh hưởng |
| --- | --- |
| Track mất trước khi tới vạch | Đếm thiếu |
| Đổi ID trước hoặc sau vạch | Vẫn đếm đúng một lần |
| Đổi ID đúng lúc vượt vạch | Có thể đếm thừa hoặc thiếu |
| Hai phương tiện che nhau, chung một ID | Đếm thiếu |
| Một phương tiện có hai khung bao khác lớp, hai ID | Có thể đếm thừa |
| Lớp thay đổi giữa chừng | Tổng không đổi, có thể sai lớp |

**Kiểm thử.** Logic đếm được kiểm thử bằng chuỗi tọa độ tổng hợp, không cần mô hình hay GPU. Các trường hợp: vượt theo cả hai chiều; không tới vạch; chạm vạch rồi quay lại; đi vòng qua đầu mút; dao động trong vùng đệm; vượt rồi quay lại; mất track ngắn và dài hơn `max_gap_frames`; đổi ID trước vạch; đổi lớp; nhiều phương tiện trong cùng khung; vạch suy biến; tính nhất quán của đầu ra. Trường hợp đổi ID đúng lúc vượt vạch được kiểm thử để ghi nhận hành vi thực tế như một giới hạn đã biết.

## 5. Định dạng đầu ra

| Tệp | Nội dung |
| --- | --- |
| `tracks.csv` | Mỗi dòng là một track trong một khung hình: `frame`, `time_s`, `track_id`, `class`, `conf`, `x1`, `y1`, `x2`, `y2`. |
| `events.csv` | Mỗi dòng là một lượt vượt vạch: `frame`, `time_s`, `track_id`, `class`. |
| `counts.json` | `grand_total`, `by_class` và cấu hình của lần chạy: video, mô hình, `conf`, tracker và tham số tracker, phiên bản thư viện, tọa độ vạch `a`, `b`, `margin_px`, `max_gap_frames`, thời gian xử lý. |

Kết quả của mỗi lần chạy nằm trong `outputs/<tên video>/` (đổi bằng `--output-dir`), gồm video, các tệp trên và `tracker.yaml`. `frame` được đánh số từ 1 và `time_s = (frame − 1) / FPS`. Tọa độ khung bao tính theo pixel của video gốc. Khung hình không có track không có dòng nào trong `tracks.csv`, và video giữ nguyên khung hình gốc.

Ràng buộc nhất quán: `grand_total` = số dòng dữ liệu của `events.csv` = tổng `by_class`. `counts.json` được tổng hợp từ chính danh sách sự kiện trong `events.csv`, và `tracks.csv` cho phép đối chiếu từng ID với video.

Video đầu ra hiển thị khung bao kèm `ID + lớp`, điểm tham chiếu, vạch đếm và bộ đếm theo loại xe ở mọi khung hình.

## 6. Tài liệu tham khảo

1. J. Redmon, S. Divvala, R. Girshick, A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection", arXiv:1506.02640, 2015.
2. A. Bewley, Z. Ge, L. Ott, F. Ramos, B. Upcroft, "Simple Online and Realtime Tracking", ICIP 2016, arXiv:1602.00763.
3. Y. Zhang, P. Sun, Y. Jiang, D. Yu, F. Weng, Z. Yuan, P. Luo, W. Liu, X. Wang, "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", arXiv:2110.06864, 2021.
4. T.-Y. Lin, M. Maire, S. Belongie, L. Bourdev, R. Girshick, J. Hays, P. Perona, D. Ramanan, C. L. Zitnick, P. Dollár, "Microsoft COCO: Common Objects in Context", arXiv:1405.0312, 2014.
5. Ultralytics, "YOLO26", https://docs.ultralytics.com/models/yolo26/
6. Code With Aarohi, "Vehicle detection and tracking classwise using YOLO11", https://github.com/codewithaarohi/Vehicle-detection-and-tracking-classwise-using-YOLO11 (tham khảo ý tưởng triển khai; không sử dụng mã nguồn hay dữ liệu).
