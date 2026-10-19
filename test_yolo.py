
from ultralytics import YOLO
import torch

# Kiểm tra GPU
device = 0 if torch.cuda.is_available() else "cpu"
print("Thiet bi xu ly:", device)

# Tải mô hình YOLO26 Nano đã huấn luyện trước
model = YOLO("yolo26n.pt")

# Nhận diện phương tiện trên ảnh mẫu
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    classes=[2, 3, 5, 7],
    conf=0.35,
    device=device,
    save=True,
    project="runs/detect",
    name="first_test"
)

print("Da hoan thanh nhan dien!")
print("Thu muc ket qua: runs/detect/first_test")
