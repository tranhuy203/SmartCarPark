import cv2
import numpy as np
import time
def count_edge_points(contour,edge_map):
    zeros_image = np.zeros_like(edge_map)
    cv2.drawContours(zeros_image, [contour], -1, (255,),thickness=-1)  # Tô trắng vùng contour
    # Áp dụng mặt nạ này lên ảnh nhị phân
    edge_map_of_contour = cv2.bitwise_and(edge_map, zeros_image)
    # Đếm số điểm ảnh trắng trong vùng này
    edge_points = cv2.countNonZero(edge_map_of_contour)
    return edge_points

def get_empty_spaces_edge_points(empty_spaces_image,contours,kernel_size = 1, sigma = 0):
    empty = []
    blurred_image = cv2.GaussianBlur(empty_spaces_image, (kernel_size, kernel_size), sigma)
    edge_map = cv2.Canny(blurred_image,50,200)
    for contour in contours:
        edge_points = count_edge_points(contour, edge_map)
        empty.append(edge_points)
    return empty
def detect_cars(cap, mask, empty_car_park_image, kernel_size = 1, sigma = 0, t_low = 50, t_high = 200, threshold_decide = 250):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    empty_spaces_edge_points = get_empty_spaces_edge_points(empty_car_park_image,contours,kernel_size, sigma)
    # tính fps
    fps_start_time = time.time()
    frame_count = 0
    fps = 0  # Khởi tạo biến fps
    while True:
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # đoc khung hinh
        _, frame = cap.read()
        blurred_image = cv2.GaussianBlur(frame, (kernel_size,kernel_size), sigma)
        edge_map = cv2.Canny(blurred_image, t_low, t_high)
        free_spaces = 0
        for i, contour in enumerate(contours):
            edge_points = count_edge_points(contour, edge_map)
            if edge_points - empty_spaces_edge_points[i] <= threshold_decide:
                free_spaces += 1
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            # Tính tọa độ trung tâm của contour (để đặt văn bản)
            m = cv2.moments(contour)
            if m["m00"] != 0:  # Tránh chia cho 0
                center_x = int(m["m10"] / m["m00"])
                center_y = int(m["m01"] / m["m00"])
            else:
                center_x, center_y = 0, 0
            # Hiển thị edge points lên ô đỗ
            cv2.putText(frame, f"{edge_points}px", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # nối các contour
            cv2.polylines(frame, [contour], isClosed=True, color=color, thickness=1)
        cv2.putText(frame, f"Free space: {free_spaces}/{len(empty_spaces_edge_points)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("result", frame)
        cv2.imshow("binary_img",edge_map)
        # Tính và hiển thị FPS
        frame_count += 1
        elapsed_time = time.time() - fps_start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            fps_start_time = time.time()
        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()
