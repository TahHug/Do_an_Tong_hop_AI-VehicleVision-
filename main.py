"""VehicleVision - Coding Task 01: phát hiện phương tiện trên video.

Đọc một video MP4, chạy YOLO26n (pretrained COCO) trên từng khung hình,
chỉ giữ 4 lớp phương tiện, vẽ khung bao + nhãn rồi ghi ra video MP4 mới.
Chưa có tracking hay đếm (các nhiệm vụ sau).

Ví dụ (PowerShell, trong thư mục dự án):
    .\\.venv\\Scripts\\python.exe main.py --video data\\videos\\dev\\day.mp4
"""

import argparse
import math
import sys
import time
from pathlib import Path

import cv2

# Chỉ số lớp trong bộ COCO mà YOLO26n pretrained đã học: 2=car, 3=motorcycle, 5=bus, 7=truck.
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

DEFAULT_MODEL = "yolo26n.pt"
DEFAULT_CONF = 0.35
FALLBACK_FPS = 30.0  # dùng khi metadata FPS của video không hợp lệ


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phát hiện car/motorcycle/bus/truck trên video bằng YOLO26n và ghi video chú thích."
    )
    parser.add_argument("--video", required=True, help="đường dẫn video đầu vào (MP4)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"tệp trọng số YOLO (mặc định: {DEFAULT_MODEL})")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help=f"ngưỡng confidence, trong khoảng (0, 1] (mặc định: {DEFAULT_CONF})")
    parser.add_argument("--output", default=None,
                        help="đường dẫn video đầu ra .mp4 (mặc định: outputs/<tên video>_detect.mp4)")
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

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("outputs") / f"{video_path.stem}_detect.mp4"
    if output_path.suffix.lower() != ".mp4":
        fail(f"Video đầu ra phải có đuôi .mp4: {output_path}")

    # Không bao giờ ghi đè video gốc.
    if output_path.resolve() == video_path.resolve():
        fail("Đường dẫn đầu ra trùng với video đầu vào; hãy chọn --output khác.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return video_path, model_path, output_path


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
    video_path, model_path, output_path = resolve_paths(args)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        fail(f"OpenCV không mở được video: {video_path}")

    writer = None
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

        frames_read = 0
        frames_written = 0
        start = time.perf_counter()
        while ok:
            frames_read += 1
            # classes=...: YOLO chỉ trả về 4 lớp xe; conf=...: bỏ khung có độ tin cậy thấp hơn ngưỡng.
            result = model.predict(frame, classes=class_ids, conf=args.conf, device=device, verbose=False)[0]
            annotated = result.plot()  # vẽ khung bao, tên lớp và confidence lên bản sao khung hình
            writer.write(annotated)
            frames_written += 1
            if frames_read % 100 == 0:
                print(f"  ... đã xử lý {frames_read} khung")
            ok, frame = capture.read()
        elapsed = time.perf_counter() - start
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    readable = count_readable_frames(output_path)
    processing_fps = frames_read / elapsed if elapsed > 0 else float("nan")

    print("Kết quả")
    print(f"  Số khung đọc  : {frames_read}")
    print(f"  Số khung ghi  : {frames_written}")
    print(f"  Thời gian     : {elapsed:.2f} s")
    print(f"  FPS xử lý     : {processing_fps:.2f} (sơ bộ, không phải phép đo chuẩn trong docs/EVALUATION.md)")
    print(f"  Video đầu ra  : {output_path.resolve()}")
    print(f"  Kiểm tra lại  : đọc được {readable} khung từ video đầu ra")

    if readable != frames_written or frames_written != frames_read:
        fail("Số khung đọc/ghi/đọc lại không khớp; video đầu ra có thể bị lỗi.")
    print("Hoàn thành.")


if __name__ == "__main__":
    main()
