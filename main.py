"""VehicleVision: phát hiện và theo dõi phương tiện trên video.

Đọc một video, chạy YOLO26n (pretrained COCO) và ByteTrack trên từng khung hình,
chỉ giữ 4 lớp phương tiện, vẽ khung bao kèm ID rồi ghi video MP4 mới và tracks.csv.
Chưa có đếm lượt vượt vạch.

Ví dụ (PowerShell, trong thư mục dự án):
    .\\.venv\\Scripts\\python.exe main.py --video data\\videos\\dev\\day.mp4
"""

import argparse
import csv
import importlib.util
import math
import os
import sys
import time
from pathlib import Path

import cv2

# Không để Ultralytics tự pip install thư viện còn thiếu (phải đặt trước khi import ultralytics).
os.environ["YOLO_AUTOINSTALL"] = "False"

# Chỉ số lớp trong bộ COCO mà YOLO26n pretrained đã học: 2=car, 3=motorcycle, 5=bus, 7=truck.
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

DEFAULT_MODEL = "yolo26n.pt"
DEFAULT_CONF = 0.35
FALLBACK_FPS = 30.0  # dùng khi metadata FPS của video không hợp lệ
CSV_COLUMNS = ["frame", "time_s", "track_id", "class", "conf", "x1", "y1", "x2", "y2"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phát hiện và theo dõi car/motorcycle/bus/truck trên video bằng YOLO26n và ByteTrack."
    )
    parser.add_argument("--video", required=True, help="đường dẫn video đầu vào")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"tệp trọng số YOLO (mặc định: {DEFAULT_MODEL})")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help=f"ngưỡng confidence để cấp ID mới (mặc định: {DEFAULT_CONF})")
    parser.add_argument("--output-dir", default=None,
                        help="thư mục kết quả (mặc định: outputs/<tên video>)")
    return parser.parse_args()


def fail(message):
    print(f"[LỖI] {message}")
    sys.exit(1)


def resolve_paths(args):
    """Kiểm tra đường dẫn trước khi nạp mô hình, để lỗi đơn giản được báo ngay."""
    video_path = Path(args.video)
    if not video_path.is_file():
        fail(f"Không tìm thấy video đầu vào: {video_path}")

    if not 0 < args.conf <= 1:
        fail(f"--conf phải nằm trong khoảng (0, 1], nhận được {args.conf}")

    model_path = Path(args.model)
    if not model_path.is_file():
        # Không để Ultralytics tự tải trọng số từ Internet: báo lỗi rõ ràng.
        fail(f"Không tìm thấy tệp trọng số: {model_path}")

    if importlib.util.find_spec("lap") is None:
        fail("Thiếu thư viện lap (ByteTrack cần lap). Cài theo mục Cài đặt trong README.md.")

    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / video_path.stem

    # Không bao giờ ghi đè video gốc.
    if (output_dir / "tracked.mp4").resolve() == video_path.resolve():
        fail("Video đầu ra trùng với video đầu vào; hãy chọn --output-dir khác.")

    output_dir.mkdir(parents=True, exist_ok=True)
    return video_path, model_path, output_dir


def write_tracker_config(output_dir, conf):
    """Tạo cấu hình ByteTrack cho lần chạy: bytetrack.yaml gốc, chỉ đổi ngưỡng cấp ID theo conf.

    ByteTrack ghép hai bước: khung bao có điểm >= track_high_thresh được ghép trước, khung có điểm
    từ track_low_thresh đến dưới track_high_thresh chỉ dùng để giữ tiếp ID đã có (ví dụ xe bị che).
    Vì vậy bộ phát hiện phải trả về cả khung điểm thấp, tức là lọc ở track_low_thresh chứ không ở conf.
    """
    from ultralytics.utils import ROOT, YAML

    config = YAML.load(ROOT / "cfg" / "trackers" / "bytetrack.yaml")
    low_thresh = config["track_low_thresh"]
    if conf <= low_thresh:
        fail(f"--conf phải lớn hơn track_low_thresh của ByteTrack ({low_thresh}), nhận được {conf}")
    config["track_high_thresh"] = conf
    config["new_track_thresh"] = conf

    tracker_path = output_dir / "tracker.yaml"
    YAML.save(tracker_path, config)
    return tracker_path, low_thresh


def extract_tracks(result):
    """Đổi kết quả một khung hình thành danh sách (track_id, tên lớp, conf, [x1, y1, x2, y2])."""
    boxes = result.boxes
    if boxes.id is None:  # khung không có track nào (các khung bao chưa có ID bị bỏ qua)
        return []
    return [
        (int(track_id), VEHICLE_CLASSES[int(class_id)], conf, xyxy)
        for track_id, class_id, conf, xyxy in zip(
            boxes.id.tolist(), boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
        )
    ]


def read_video_info(capture):
    """Lấy FPS, kích thước, số khung từ metadata; FPS sai thì dùng giá trị dự phòng."""
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # có thể 0 hoặc ước lượng

    fps_note = ""
    if not fps or math.isnan(fps) or fps <= 0 or fps > 240:
        fps_note = f" (metadata FPS không hợp lệ: {fps!r}, dùng {FALLBACK_FPS})"
        fps = FALLBACK_FPS
    return fps, width, height, frame_count, fps_note


def choose_device():
    import torch

    if torch.cuda.is_available():
        return 0, f"CUDA ({torch.cuda.get_device_name(0)})"
    return "cpu", "CPU"


def load_model(model_path):
    from ultralytics import YOLO  # nạp sau khi đã kiểm tra đường dẫn

    try:
        return YOLO(str(model_path))
    except Exception as error:
        fail(f"Không nạp được mô hình {model_path}: {error}")


def count_readable_frames(path):
    """Mở lại video đã ghi và đếm số khung đọc được, để không báo thành công với tệp hỏng."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return 0
    count = 0
    while capture.read()[0]:
        count += 1
    capture.release()
    return count


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # tránh lỗi in tiếng Việt khi terminal không dùng UTF-8

    args = parse_args()
    video_path, model_path, output_dir = resolve_paths(args)
    tracker_path, low_thresh = write_tracker_config(output_dir, args.conf)
    output_path = output_dir / "tracked.mp4"
    csv_path = output_dir / "tracks.csv"

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        fail(f"OpenCV không mở được video: {video_path}")

    writer = None
    csv_file = None
    try:
        fps, width, height, meta_frames, fps_note = read_video_info(capture)

        # Đọc khung đầu tiên để chắc chắn video giải mã được và biết kích thước thật.
        ok, frame = capture.read()
        if not ok:
            fail(f"Không đọc được khung hình nào từ video: {video_path}")
        height, width = frame.shape[:2]

        print(f"Video đầu vào : {video_path}")
        print(f"  Kích thước  : {width}x{height}, FPS {fps:.2f}{fps_note}, "
              f"số khung theo metadata: {meta_frames if meta_frames > 0 else 'không rõ'}")

        device, device_name = choose_device()
        print(f"Thiết bị      : {device_name}")
        model = load_model(model_path)
        class_ids = list(VEHICLE_CLASSES)

        # Mở VideoWriter sau khi nạp mô hình xong, để lỗi mô hình không để lại tệp đầu ra rỗng.
        # mp4v: codec MPEG-4 Part 2 mà OpenCV ghi được vào tệp .mp4 trên Windows.
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            fail(f"Không mở được VideoWriter cho tệp: {output_path}")

        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(CSV_COLUMNS)

        frames_read = 0
        frames_written = 0
        track_rows = 0
        track_ids = set()
        start = time.perf_counter()
        while ok:
            frames_read += 1
            # persist=True: giữ trạng thái tracker giữa các khung. conf=low_thresh: ByteTrack cần cả khung
            # điểm thấp; ngưỡng cấp ID mới (--conf) nằm trong tracker.yaml. agnostic_nms=True: mỗi vị trí dự
            # đoán chỉ giữ một lớp, tránh một xe có hai khung bao trùng nhau (vd. car và truck) thành hai ID.
            result = model.track(frame, persist=True, tracker=str(tracker_path), classes=class_ids,
                                 conf=low_thresh, agnostic_nms=True, device=device, verbose=False)[0]
            tracks = extract_tracks(result)
            time_s = (frames_read - 1) / fps  # frame đánh số từ 1
            for track_id, class_name, conf, xyxy in tracks:
                csv_writer.writerow([frames_read, f"{time_s:.3f}", track_id, class_name, f"{conf:.4f}",
                                     *(f"{value:.1f}" for value in xyxy)])
                track_ids.add(track_id)
            track_rows += len(tracks)
            # plot() vẽ "id:<ID> <lớp> <conf>". Không có track thì giữ khung gốc, tránh vẽ khung bao không ID.
            writer.write(result.plot() if tracks else frame)
            frames_written += 1
            if frames_read % 100 == 0:
                print(f"  ... đã xử lý {frames_read} khung")
            ok, frame = capture.read()
        elapsed = time.perf_counter() - start
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if csv_file is not None:
            csv_file.close()

    readable = count_readable_frames(output_path)
    processing_fps = frames_read / elapsed if elapsed > 0 else float("nan")

    print("Kết quả")
    print(f"  Số khung đọc  : {frames_read}")
    print(f"  Số khung ghi  : {frames_written}")
    print(f"  Thời gian     : {elapsed:.2f} s")
    print(f"  FPS xử lý     : {processing_fps:.2f} (sơ bộ, không phải phép đo chuẩn trong docs/EVALUATION.md)")
    print(f"  Số dòng track : {track_rows}")
    print(f"  Số ID khác nhau: {len(track_ids)} (không phải số xe đã đếm)")
    print(f"  Video đầu ra  : {output_path.resolve()}")
    print(f"  tracks.csv    : {csv_path.resolve()}")
    print(f"  Cấu hình track: {tracker_path.resolve()}")
    print(f"  Kiểm tra lại  : đọc được {readable} khung từ video đầu ra")

    if readable != frames_written or frames_written != frames_read:
        fail("Số khung đọc/ghi/đọc lại không khớp; video đầu ra có thể bị lỗi.")
    print("Hoàn thành.")


if __name__ == "__main__":
    main()
