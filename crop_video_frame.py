import cv2

video_path = 'data/carPark2.mp4'
output_path = 'data/carPark2.mp4'

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 🪄 Xác định vùng crop (ví dụ crop ở giữa khung hình)
crop_x = 26  # điểm bắt đầu theo chiều ngang
crop_y = 11   # điểm bắt đầu theo chiều dọc
crop_width = 1286
crop_height = 780

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (crop_width, crop_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ✂️ Crop frame
    cropped_frame = frame[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]

    out.write(cropped_frame)

cap.release()
out.release()
print('✅ Đã crop video thành công!')