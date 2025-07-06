import cv2
import time
fps_start_time = time.time()
frame_count = 0
fps = 0  # Khởi tạo biến fps
cap = cv2.VideoCapture("data/carPark2.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    # Nhấn phím 'q' để thoát
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    cv2.putText(frame, f"FPS: {fps:.2f}", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Tính và hiển thị FPS
    # Hiển thị frame
    cv2.imshow('Video', frame)
    frame_count += 1
    elapsed_time = time.time() - fps_start_time
    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        frame_count = 0
        fps_start_time = time.time()
# Giải phóng bộ nhớ và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()