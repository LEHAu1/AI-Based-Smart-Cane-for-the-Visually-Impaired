# AI-Based Smart Cane for the Visually Impaired 🚶‍♂️🦯

Hệ thống AI hỗ trợ người khiếm thị nhận diện vật cản thông qua Camera và cảnh báo âm thanh không dây bằng mạch ESP32.

## 🌟 Tính năng chính & Kiến trúc hệ thống
Dựa trên mã nguồn `ai_server.py`, hệ thống hoạt động với kiến trúc như sau:
- **Xử lý Hình ảnh (Vision):** Sử dụng trực tiếp Camera kết nối với máy tính/máy chủ (`Webcam`). Khung hình được phân tích liên tục bằng mô hình **YOLOv8** (`yolov8n.pt` và `best.pt`) và mô hình ước lượng chiều sâu **Depth-Anything-V2-Small** (HuggingFace) để phát hiện xe cộ, người, vạch qua đường, hố sâu và bậc thang.
- **Phản hồi Giọng nói (Text-to-Speech):** Khi phát hiện nguy hiểm, AI Server sử dụng Google Text-to-Speech (`gTTS`) để tạo ra câu cảnh báo ngắn gọn bằng tiếng Việt (ví dụ: *"Cảnh báo! Có hố phía trước"*, *"Xe bên trái"*).
- **Truyền Audio không dây (Raw TCP Socket):** Server tự động chuyển đổi âm thanh sang định dạng PCM 16kHz, tăng âm lượng, rồi stream liên tục qua cổng TCP (Port `5000`).
- **Phát âm thanh trên ESP32 (I2S):** Mạch ESP32 đóng vai trò là một "loa không dây", chủ động kết nối Wi-Fi tới Server TCP, nhận gói tin Audio và phát thẳng ra loa ngoài qua module Amply I2S (MAX98357A). 

## 🛠️ Yêu cầu phần cứng
- 1 x Máy tính/Laptop có tích hợp hoặc cắm thêm Camera (Webcam).
- 1 x Mạch ESP32 (hoặc ESP32-CAM).
- 1 x Module khuếch đại âm thanh I2S MAX98357A.
- 1 x Loa nhỏ 4-8 Ohm (3W).
- Nguồn cấp cho mạch ESP32 (Pin dự phòng hoặc Lipo).

## ⚙️ Cài đặt & Khởi chạy hệ thống

### Bước 1: Khởi chạy AI Server (Trên Máy tính)
1. Clone dự án về máy:
   ```bash
   git clone https://github.com/LEHAu1/AI-Based-Smart-Cane-for-the-Visually-Impaired.git
   cd AI-Based-Smart-Cane-for-the-Visually-Impaired
   ```
2. Tạo và kích hoạt môi trường ảo:
   ```bash
   python -m venv .venv
   
   # Trên Windows:
   .venv\Scripts\activate
   # Trên MacOS/Linux:
   source .venv/bin/activate
   ```
3. Cài đặt các thư viện bắt buộc (PyTorch, Ultralytics, Transformers, OpenCV, gTTS...):
   ```bash
   pip install -r requirements.txt
   ```
4. Khởi chạy AI Server:
   ```bash
   python ai_server.py
   ```
   *Lưu ý: Màn hình camera sẽ hiện ra kèm các khung chỉ định vật cản nhận diện được. Server sẽ mở cổng TCP 5000 chờ ESP32 kết nối.*
   *Hãy dùng lệnh `ipconfig` (Windows) hoặc `ifconfig` (Mac/Linux) để xem địa chỉ IP Local của máy tính (ví dụ: `192.168.1.15`), ghi nhớ IP này cho Bước 2.*

### Bước 2: Nạp code cho Mạch phát âm thanh (ESP32)
1. Mở phần mềm `Arduino IDE` và nhớ cài đặt các thư viện I2S Audio cần thiết cho mạch ESP32.
2. Mở file `esp32_audio_AI_cam.ino`.
3. Tìm phần khai báo Wi-Fi & IP, thay đổi cho khớp với mạng nhà bạn:
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";          // Tên mạng Wi-Fi
   const char* password = "YOUR_WIFI_PASSWORD";  // Mật khẩu mạng Wi-Fi
   
   // >>>> THAY ĐỊA CHỈ IP MÁY TÍNH CỦA BẠN VÀO ĐÂY <<<<
   const char* host = "192.168.1.15";  
   const int port = 5000;
   ```
4. Đấu nối các chân từ ESP32 sang MAX98357A theo đúng chuẩn I2S (BCLK, LRC, DIN) khai báo trong code Arduino.
5. Cắm cáp và nhấn nút **Upload** nạp chương trình vào mạch ESP32.

## 🤝 Đóng góp
Dự án được thực hiện với các mục tiêu giáo dục và hỗ trợ cộng đồng. Mọi ý kiến đóng góp, báo lỗi cũng như Pull Requests đều được hoan nghênh nồng nhiệt! Trang thái kết nối và Ping của ESP32 có thể xem realtime trực tiếp trên màn hình camera hiển thị của Server.

---
*Dự án phát triển bởi nhóm tác giả @LEHAu1.*
