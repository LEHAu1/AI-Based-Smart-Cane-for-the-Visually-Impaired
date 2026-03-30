import cv2
import numpy as np
import socket
import time
import threading
import queue
import tempfile
import os
import torch
from collections import defaultdict
from gtts import gTTS
import soundfile as sf
import scipy.signal as sps
from ultralytics import YOLO
from transformers import pipeline

# ═══════════════════════════════════════════════════════
# CẤU HÌNH
# ═══════════════════════════════════════════════════════
ESP32_IP    = "192.168.1.7"
ESP32_PORT  = 5000
SAMPLE_RATE = 16000
FRAME_W, FRAME_H = 640, 480

VEHICLE_LABELS = {"car", "motorcycle", "bus", "truck"}

# Cooldown (giây) theo loại nguy hiểm
COOLDOWN = {
    "vehicle_near": 4,
    "pothole":      4,
    "stairs":       7,
    "traffic_red":  3,
    "person":       3,
    "zebra":        5,
    "vehicle_mid":  5,
}

# Số frame liên tiếp cần detect trước khi cảnh báo
CONFIRM_FRAMES = {
    "pothole":      4,
    "stairs":       4,
    "zebra":        3,
    "person":       2,
    "vehicle_near": 2,
    "vehicle_mid":  2,
    "traffic_red":  3,
}

# Ngưỡng bbox/frame ratio để phân loại khoảng cách xe
VEHICLE_NEAR_RATIO = 0.18
VEHICLE_MID_RATIO  = 0.06


# ═══════════════════════════════════════════════════════
# ALERT MODULE
# ═══════════════════════════════════════════════════════

def get_priority(obj: dict) -> int:
    """Trả về mức ưu tiên (số nhỏ = quan trọng hơn)."""
    t = obj["type"]
    if t in ("vehicle_near", "pothole", "stairs"):
        return 1
    if t in ("traffic_red", "person"):
        return 2
    return 3

def get_distance_level(bbox_ratio: float) -> str:
    """Phân loại khoảng cách dựa trên tỉ lệ diện tích bbox / frame."""
    if bbox_ratio > VEHICLE_NEAR_RATIO:
        return "very_close"
    if bbox_ratio > VEHICLE_MID_RATIO:
        return "medium"
    return "far"

def build_message(obj: dict) -> str:
    """Tạo câu cảnh báo ngắn gọn bằng tiếng Việt."""
    t   = obj["type"]
    d   = obj["direction"]
    lvl = obj.get("dist_level", "medium")
    
    prefix = "Cảnh báo! " if lvl == "very_close" or t in ("pothole", "stairs") else ""
    
    messages = {
        "vehicle_near": f"{prefix}Xe {d}",
        "vehicle_mid":  f"Có xe {d}",
        "pothole":      f"{prefix}Có hố {d}",
        "stairs":       f"{prefix}Có bậc thang {d}",
        "traffic_red":  "Đèn đỏ, dừng lại",
        "person":       f"Có người {d}",
        "zebra":        f"Có vạch qua đường {d}",
    }
    return messages.get(t, "")

_cooldown_lock = threading.Lock()
_last_alert_time: dict = defaultdict(float)
_esp32_client = None
_esp32_lock   = threading.Lock()
_is_playing   = threading.Event()

# Ping/latency tracking
_ping_ms = 0
_last_ping_time = 0
_ping_lock = threading.Lock()

def measure_ping():
    """Đo ping đến ESP32."""
    global _ping_ms, _last_ping_time
    with _esp32_lock:
        if _esp32_client is None:
            with _ping_lock:
                _ping_ms = -1
            print("[PING] ESP32 not connected")
            return
        client = _esp32_client
    
    try:
        start = time.time()
        # Gửi 1 byte test
        client.setblocking(False)
        client.recv(1, socket.MSG_PEEK)
        client.setblocking(True)
        elapsed = (time.time() - start) * 1000
        with _ping_lock:
            _ping_ms = int(elapsed)
            _last_ping_time = time.time()
        
        # Log ping result
        if elapsed < 50:
            status = "EXCELLENT"
        elif elapsed < 100:
            status = "GOOD"
        elif elapsed < 200:
            status = "FAIR"
        else:
            status = "POOR"
        print(f"[PING] {int(elapsed)}ms - {status}")
        
    except BlockingIOError:
        # Không có data nhưng connection OK
        elapsed = (time.time() - start) * 1000
        with _ping_lock:
            _ping_ms = int(elapsed)
            _last_ping_time = time.time()
        print(f"[PING] {int(elapsed)}ms (no data)")
    except Exception as e:
        with _ping_lock:
            _ping_ms = -1
        print(f"[PING] ERROR: {e}")

def get_ping_status():
    """Trả về (ping_ms, status_text, color)."""
    with _ping_lock:
        ping = _ping_ms
        age = time.time() - _last_ping_time
    
    if ping < 0 or age > 5:
        status = "OFFLINE"
        color = (0, 0, 255)
        bars = 0
    elif ping < 50:
        status = "EXCELLENT"
        color = (0, 255, 0)
        bars = 4
    elif ping < 100:
        status = "GOOD"
        color = (0, 200, 100)
        bars = 3
    elif ping < 200:
        status = "FAIR"
        color = (0, 165, 255)
        bars = 2
    else:
        status = "POOR"
        color = (0, 100, 255)
        bars = 1
    
    # Log status periodically (every 5 seconds)
    if hasattr(get_ping_status, '_last_log'):
        if time.time() - get_ping_status._last_log > 5:
            print(f"[STATUS] Ping: {ping}ms | Status: {status} | Bars: {bars}/4 | Age: {age:.1f}s")
            get_ping_status._last_log = time.time()
    else:
        get_ping_status._last_log = time.time()
    
    return ping, status, color

def should_alert(obj: dict) -> bool:
    """Kiểm tra cooldown. Trả về True nếu được phép phát cảnh báo."""
    with _esp32_lock:
        if _esp32_client is None:
            return False
    if _is_playing.is_set():
        return False
    key = obj["cooldown_key"]
    cd  = COOLDOWN.get(key, 4)
    now = time.time()
    with _cooldown_lock:
        return now - _last_alert_time[key] >= cd

def mark_alert_sent(cooldown_key: str):
    """Đánh dấu đã phát cảnh báo."""
    now = time.time()
    with _cooldown_lock:
        _last_alert_time[cooldown_key] = now


def pick_top_alert(detections: list) -> dict:
    """Chọn 1 cảnh báo quan trọng nhất."""
    dist_order = {"very_close": 0, "medium": 1, "far": 2}
    detections.sort(key=lambda o: (
        get_priority(o),
        dist_order.get(o.get("dist_level", "medium"), 1)
    ))
    for obj in detections:
        if should_alert(obj):
            return obj
    return {}

# ═══════════════════════════════════════════════════════
# AUDIO
# ═══════════════════════════════════════════════════════

_tts_cache: dict = {}

def text_to_pcm(text: str, lang: str = "vi") -> bytes:
    """Chuyển text → PCM 16kHz mono. Cache theo text gốc."""
    cache_key = text.strip()
    if cache_key in _tts_cache:
        return _tts_cache[cache_key]
    
    words = text.split()
    dedup = [words[0]] if words else []
    for w in words[1:]:
        if w.lower() != dedup[-1].lower():
            dedup.append(w)
    text = " ".join(dedup)
    if text and text[-1] not in ".!?,":
        text += "."
    
    print(f"[TTS] Generating: '{text}'")
    tts = gTTS(text=text, lang=lang, slow=False)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name
        tts.save(tmp_path)
    try:
        data, orig_sr = sf.read(tmp_path, dtype="int16", always_2d=False)
        if orig_sr != SAMPLE_RATE:
            n = int(len(data) * SAMPLE_RATE / orig_sr)
            data = sps.resample(data, n).astype("int16")
        if data.ndim > 1:
            data = data[:, 0]
        nonsilent = np.where(np.abs(data) > 300)[0]
        if nonsilent.size:
            data = data[:nonsilent[-1] + 1]
        
        # TĂNG VOLUME - giảm để tránh distortion
        VOLUME_GAIN = 1.5
        data = data.astype(np.float32) * VOLUME_GAIN
        data = np.clip(data, -32768, 32767).astype(np.int16)
        
        data = np.concatenate([data, np.zeros(int(SAMPLE_RATE * 0.25), dtype="int16")])
    finally:
        os.unlink(tmp_path)
    
    pcm = data.tobytes()
    _tts_cache[cache_key] = pcm
    return pcm

audio_queue: queue.Queue = queue.Queue(maxsize=1)


def _tcp_server_thread():
    global _esp32_client
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", ESP32_PORT))
    srv.listen(1)
    print(f"[Audio] Server listening :{ESP32_PORT} — waiting for ESP32...")
    
    def _monitor_connection(conn, addr):
        global _esp32_client
        print(f"[Monitor] Started for {addr[0]}")
        try:
            while True:
                time.sleep(2)  # Ping mỗi 2 giây
                with _esp32_lock:
                    if _esp32_client != conn:
                        print(f"[Monitor] Client changed, stopping monitor for {addr[0]}")
                        break
                
                # Đo ping
                print(f"[Monitor] Measuring ping to {addr[0]}...")
                measure_ping()
                
                try:
                    conn.setblocking(False)
                    data = conn.recv(1, socket.MSG_PEEK)
                    conn.setblocking(True)
                    if not data and conn.fileno() == -1:
                        raise ConnectionError("Connection closed")
                except BlockingIOError:
                    conn.setblocking(True)
                    pass
                except Exception as e:
                    print(f"[Monitor] Connection lost: {e}")
                    break
        finally:
            with _esp32_lock:
                if _esp32_client == conn:
                    try: _esp32_client.close()
                    except: pass
                    _esp32_client = None
            print(f"[Monitor] Stopped for {addr[0]}")
    
    while True:
        try:
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            if hasattr(socket, 'TCP_KEEPIDLE'):
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 10)
            if hasattr(socket, 'TCP_KEEPINTVL'):
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 5)
            if hasattr(socket, 'TCP_KEEPCNT'):
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
            conn.settimeout(None)
            
            with _esp32_lock:
                if _esp32_client:
                    try: _esp32_client.close()
                    except: pass
                _esp32_client = conn
            
            print(f"[Audio] ESP32 connected from {addr[0]}")
            t = np.linspace(0, 0.3, int(SAMPLE_RATE * 0.3), dtype=np.float32)
            beep = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
            try: conn.sendall(beep.tobytes())
            except: pass
            
            threading.Thread(target=_monitor_connection, args=(conn, addr), 
                           daemon=True, name=f"Monitor-{addr[0]}").start()
        except Exception as e:
            print(f"[Audio] Server accept error: {e}")
            time.sleep(1)

threading.Thread(target=_tcp_server_thread, daemon=True, name="TCPServer").start()


def _audio_worker():
    global _esp32_client
    print("[Audio] Worker thread started")
    while True:
        pcm_bytes = audio_queue.get()
        if pcm_bytes is None:
            print("[Audio] Worker thread stopping")
            break
        
        print(f"[Audio] Worker got {len(pcm_bytes)} bytes from queue")
        max_retries = 3
        for attempt in range(max_retries):
            with _esp32_lock:
                client = _esp32_client
            
            if client is None:
                print(f"[Audio] No client (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
                else:
                    print("[Audio] Giving up - no client available")
                    audio_queue.task_done()
                    break
            
            _is_playing.set()
            print(f"[Audio] Sending {len(pcm_bytes)} bytes to ESP32 (attempt {attempt+1})...")
            try:
                client.sendall(pcm_bytes)
                print(f"[Audio] Send successful!")
                break
            except Exception as e:
                print(f"[Audio] Send error (attempt {attempt+1}/{max_retries}): {e}")
                with _esp32_lock:
                    if _esp32_client == client:
                        try: _esp32_client.close()
                        except: pass
                        _esp32_client = None
                if attempt < max_retries - 1:
                    time.sleep(0.5)
            finally:
                _is_playing.clear()
        
        audio_queue.task_done()

threading.Thread(target=_audio_worker, daemon=True, name="AudioWorker").start()

def speak(text: str, cooldown_key: str = None):
    """Queue audio. Trả về True nếu thành công."""
    if not text:
        print("[Audio] speak() called with empty text")
        return False
    
    with _esp32_lock:
        if _esp32_client is None:
            print(f"[Audio] Cannot queue '{text}' - ESP32 not connected")
            return False
    
    if _is_playing.is_set():
        print(f"[Audio] Cannot queue '{text}' - already playing")
        return False
    
    dropped = 0
    while not audio_queue.empty():
        try: 
            audio_queue.get_nowait()
            dropped += 1
        except: break
    
    if dropped > 0:
        print(f"[Audio] Dropped {dropped} old audio(s) from queue")
    
    try:
        pcm_data = text_to_pcm(text)
        print(f"[Audio] Generated PCM: {len(pcm_data)} bytes for '{text}'")
        audio_queue.put_nowait(pcm_data)
        print(f"[Audio] Queued successfully: '{text}'")
        
        if cooldown_key:
            mark_alert_sent(cooldown_key)
            print(f"[Audio] Cooldown marked for: {cooldown_key}")
        
        return True
    except queue.Full:
        print(f"[Audio] Queue full, cannot add: '{text}'")
        return False
    except Exception as e:
        print(f"[Audio] Error in speak(): {e}")
        return False


# ═══════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════

def get_direction(cx: int, frame_width: int) -> str:
    """Trả về hướng bằng tiếng Việt cho TTS."""
    r = cx / frame_width
    if r < 0.33:  return "bên trái"
    if r > 0.67:  return "bên phải"
    return "phía trước"

def get_direction_en(cx: int, frame_width: int) -> str:
    """Trả về hướng bằng tiếng Anh cho HUD."""
    r = cx / frame_width
    if r < 0.33:  return "left"
    if r > 0.67:  return "right"
    return "ahead"

def _classify_traffic_light(roi: np.ndarray, current: str) -> str:
    """Phân loại màu đèn giao thông."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_h, roi_w = roi.shape[:2]
    total_pixels = roi_h * roi_w
    
    red_lo = cv2.inRange(hsv, (0, 100, 70), (10, 255, 255))
    red_hi = cv2.inRange(hsv, (160, 100, 70), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_lo, red_hi)
    green_mask = cv2.inRange(hsv, (40, 70, 70), (90, 255, 255))
    yellow_mask = cv2.inRange(hsv, (15, 100, 100), (35, 255, 255))
    
    threshold = max(30, total_pixels * 0.03)
    red_count = cv2.countNonZero(red_mask)
    green_count = cv2.countNonZero(green_mask)
    yellow_count = cv2.countNonZero(yellow_mask)
    
    if current == "RED":
        return "RED"
    if red_count > threshold and red_count >= green_count:
        return "RED"
    if green_count > threshold and green_count > yellow_count:
        return "GREEN"
    return current

def detect_pothole_depth(depth_bottom: np.ndarray) -> bool:
    """Phát hiện hố từ depth map."""
    d = depth_bottom.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    blackhat = cv2.morphologyEx(d, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 0.12, 1.0, cv2.THRESH_BINARY)
    mask = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fh, fw = depth_bottom.shape[:2]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 600: continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        if y + ch < fh * 0.2: continue
        p = cv2.arcLength(cnt, True)
        if p == 0: continue
        if 4 * np.pi * area / (p ** 2) > 0.2:
            return True
    return False

def detect_stairs_depth(depth_bottom: np.ndarray) -> bool:
    """Phát hiện bậc thang từ depth map."""
    from scipy.signal import find_peaks
    d = depth_bottom.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)
    grad_y = np.abs(cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3))
    profile = grad_y.mean(axis=1)
    peaks, _ = find_peaks(profile, height=profile.mean() * 2.0, distance=8)
    if len(peaks) < 4: return False
    gaps = np.diff(peaks)
    avg = gaps.mean()
    return float(avg) > 5 and bool(np.all(np.abs(gaps - avg) < avg * 0.5))

def detect_zebra(frame_half: np.ndarray) -> bool:
    """Phát hiện vạch qua đường."""
    gray = cv2.cvtColor(frame_half, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                           minLineLength=60, maxLineGap=10)
    if lines is None: return False
    h_ys = sorted(ly1 for line in lines
                  for lx1, ly1, lx2, ly2 in [line[0]] if abs(ly1 - ly2) < 8)
    if len(h_ys) < 6: return False
    gaps = [h_ys[i+1] - h_ys[i] for i in range(len(h_ys) - 1)]
    avg = sum(gaps) / len(gaps)
    return avg > 3 and all(abs(g - avg) < avg * 0.6 for g in gaps)


# ═══════════════════════════════════════════════════════
# INFERENCE WORKER
# ═══════════════════════════════════════════════════════

class InferenceWorker:
    """YOLO + Depth chạy trên thread riêng."""
    
    def __init__(self, model, depth_pipe, hazard_model=None):
        self.model = model
        self.depth_pipe = depth_pipe
        self.hazard_model = hazard_model
        self._frame_queue = queue.Queue(maxsize=1)
        self._result_lock = threading.Lock()
        self._latest: dict = {
            "boxes": [],
            "traffic": "NONE",
            "pothole": False,
            "stairs": False,
            "zebra": False,
            "hazard_boxes": [],
        }
        self._confirm: dict = defaultdict(int)
        self._running = True
        threading.Thread(target=self._run, daemon=True, name="InferenceWorker").start()
    
    def submit(self, frame: np.ndarray):
        try:
            self._frame_queue.get_nowait()
        except queue.Empty:
            pass
        self._frame_queue.put(frame.copy())
    
    @property
    def results(self) -> dict:
        with self._result_lock:
            return dict(self._latest)
    
    def stop(self):
        self._running = False
        try: self._frame_queue.put_nowait(None)
        except queue.Full: pass
    
    def _confirm_key(self, key: str, detected: bool) -> bool:
        if detected:
            self._confirm[key] += 1
        else:
            self._confirm[key] = 0
        return self._confirm[key] >= CONFIRM_FRAMES.get(key, 1)
    
    def _run(self):
        while self._running:
            frame = self._frame_queue.get()
            if frame is None: break
            h, w = frame.shape[:2]
            
            yolo_res = self.model(frame, verbose=False)
            traffic = "NONE"
            boxes = []
            
            for r in yolo_res:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = self.model.names[cls]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    if label == "traffic light":
                        roi = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                        if roi.size >= 100:
                            traffic = _classify_traffic_light(roi, traffic)
                    elif label in VEHICLE_LABELS or label == "person":
                        boxes.append((label, x1, y1, x2, y2, conf))
            
            pothole = False
            stairs = False
            zebra = False
            hazard_boxes = []
            
            if self.hazard_model is not None:
                hres = self.hazard_model(frame, verbose=False, conf=0.35)
                for r in hres:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = self.hazard_model.names[cls]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        if label == "pothole" and conf >= 0.5: pothole = True
                        elif label == "stairs" and conf >= 0.6: stairs = True
                        elif label == "zebra_crossing" and conf >= 0.35: zebra = True
                        hazard_boxes.append((label, x1, y1, x2, y2, conf))
            else:
                zebra = detect_zebra(frame[h//2:])
                small = cv2.resize(frame, (320, 240))
                from PIL import Image as PILImage
                depth_out = self.depth_pipe(PILImage.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
                depth_map = np.array(depth_out["depth"], dtype=np.float32)
                depth_bottom = depth_map[depth_map.shape[0]//2:]
                pothole = detect_pothole_depth(depth_bottom)
                stairs = detect_stairs_depth(depth_bottom)
            
            with self._result_lock:
                self._latest.update({
                    "boxes": boxes,
                    "traffic": "RED" if self._confirm_key("traffic_red", traffic == "RED") else
                              ("GREEN" if traffic == "GREEN" else "NONE"),
                    "pothole": self._confirm_key("pothole", pothole),
                    "stairs": self._confirm_key("stairs", stairs),
                    "zebra": self._confirm_key("zebra", zebra),
                    "hazard_boxes": hazard_boxes,
                })


# ═══════════════════════════════════════════════════════
# BUILD DETECTIONS
# ═══════════════════════════════════════════════════════

def build_detections(res: dict, frame_w: int, frame_h: int) -> list:
    """Chuyển kết quả inference thành list detection objects."""
    detections = []
    frame_area = frame_w * frame_h
    
    for (label, x1, y1, x2, y2, conf) in res["boxes"]:
        cx = (x1 + x2) // 2
        direction = get_direction(cx, frame_w)
        ratio = (x2 - x1) * (y2 - y1) / frame_area
        dist_lvl = get_distance_level(ratio)
        
        if label in VEHICLE_LABELS:
            if dist_lvl == "far":
                continue
            obj_type = "vehicle_near" if dist_lvl == "very_close" else "vehicle_mid"
            detections.append({
                "type": obj_type,
                "direction": direction,
                "dist_level": dist_lvl,
                "priority": get_priority({"type": obj_type}),
                "cooldown_key": obj_type,
                "ratio": ratio,
            })
        elif label == "person":
            if dist_lvl == "far":
                continue
            detections.append({
                "type": "person",
                "direction": direction,
                "dist_level": dist_lvl,
                "priority": 2,
                "cooldown_key": "person",
                "ratio": ratio,
            })
    
    if res["traffic"] == "RED":
        detections.append({
            "type": "traffic_red",
            "direction": "phía trước",
            "dist_level": "medium",
            "priority": 2,
            "cooldown_key": "traffic_red",
        })
    
    if res["pothole"]:
        detections.append({
            "type": "pothole",
            "direction": "phía trước",
            "dist_level": "very_close",
            "priority": 1,
            "cooldown_key": "pothole",
        })
    
    if res["stairs"]:
        detections.append({
            "type": "stairs",
            "direction": "phía trước",
            "dist_level": "very_close",
            "priority": 1,
            "cooldown_key": "stairs",
        })
    
    if res["zebra"]:
        detections.append({
            "type": "zebra",
            "direction": "phía trước",
            "dist_level": "medium",
            "priority": 3,
            "cooldown_key": "zebra",
        })
    
    return detections


# ═══════════════════════════════════════════════════════
# DRAW HUD
# ═══════════════════════════════════════════════════════

def draw_hud(frame: np.ndarray, res: dict, detections: list, top_alert: dict):
    """Vẽ HUD lên frame - text tiếng Anh để tránh lỗi font."""
    h, w = frame.shape[:2]
    
    # ESP32 Connection Status với ping/latency (top-right corner)
    ping_ms, status, color = get_ping_status()
    
    if ping_ms >= 0:
        status_text = f"ESP32: {ping_ms}ms"
        status_sub = status
    else:
        status_text = "ESP32: OFFLINE"
        status_sub = "No Connection"
    
    # Draw background box
    text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    sub_size = cv2.getTextSize(status_sub, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    box_width = max(text_size[0], sub_size[0]) + 50
    box_x1 = w - box_width - 10
    box_y1 = 5
    box_x2 = w - 5
    box_y2 = 55
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), color, 2)
    
    # Draw signal strength bars
    bar_x = box_x1 + 10
    bar_y = box_y1 + 15
    bar_count = 4
    if ping_ms < 0:
        bars_active = 0
    elif ping_ms < 50:
        bars_active = 4
    elif ping_ms < 100:
        bars_active = 3
    elif ping_ms < 200:
        bars_active = 2
    else:
        bars_active = 1
    
    for i in range(bar_count):
        bar_height = 5 + i * 4
        bar_color = color if i < bars_active else (100, 100, 100)
        cv2.rectangle(frame, 
                     (bar_x + i * 6, bar_y + (bar_count * 4 - bar_height)),
                     (bar_x + i * 6 + 4, bar_y + bar_count * 4),
                     bar_color, -1)
    
    # Draw status text
    cv2.putText(frame, status_text, (bar_x + 30, box_y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, status_sub, (bar_x + 30, box_y1 + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    for (label, x1, y1, x2, y2, conf) in res["boxes"]:
        cx = (x1 + x2) // 2
        ratio = (x2-x1)*(y2-y1) / (h*w)
        if label in VEHICLE_LABELS:
            color = (0,0,255) if ratio > VEHICLE_NEAR_RATIO else (0,165,255)
            cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
            cv2.putText(frame, f"{label} {get_direction_en(cx,w)}", (x1, max(y1-8,10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        elif label == "person":
            cv2.rectangle(frame, (x1,y1),(x2,y2),(255,200,0),2)
            cv2.putText(frame, f"person {get_direction_en(cx,w)}", (x1, max(y1-8,10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,200,0), 2)
    
    for (label, x1, y1, x2, y2, conf) in res["hazard_boxes"]:
        color = (0,0,200) if label == "pothole" else \
                (255,128,0) if label == "stairs" else (0,255,255)
        cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1-8,10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    
    y = 30
    traffic_color = (0,0,255) if res["traffic"] == "RED" else \
                    (0,200,0) if res["traffic"] == "GREEN" else (150,150,150)
    cv2.putText(frame, f"TRAFFIC: {res['traffic']}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, traffic_color, 2); y += 32
    
    flags = []
    if res["pothole"]: flags.append("POTHOLE")
    if res["stairs"]: flags.append("STAIRS")
    if res["zebra"]: flags.append("ZEBRA CROSSING")
    if flags:
        cv2.putText(frame, " | ".join(flags), (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
    y += 32
    
    if top_alert:
        t = top_alert["type"]
        d = top_alert.get("direction", "ahead")
        d_en = "left" if "trái" in d else "right" if "phải" in d else "ahead"
        eng_messages = {
            "vehicle_near": f"DANGER! Vehicle {d_en}",
            "vehicle_mid": f"Vehicle {d_en}",
            "pothole": f"DANGER! Pothole {d_en}",
            "stairs": f"DANGER! Stairs {d_en}",
            "traffic_red": "RED LIGHT - STOP",
            "person": f"Person {d_en}",
            "zebra": f"Zebra crossing {d_en}",
        }
        msg = eng_messages.get(t, "")
        cv2.putText(frame, f">> {msg}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    else:
        cv2.putText(frame, "NO_ALERT", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,100), 1)


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    print("[YOLO] Loading YOLOv8n...")
    model = YOLO("yolov8n.pt")
    
    print("[DEPTH] Loading Depth-Anything-V2-Small...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    depth_pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=device,
    )
    print(f"[DEPTH] Ready (device={device})")
    
    hazard_model = None
    if os.path.exists("best.pt"):
        print("[HAZARD] Loading best.pt...")
        hazard_model = YOLO("best.pt")
        print("[HAZARD] Ready")
    else:
        print("[HAZARD] best.pt not found → using Depth-Anything fallback")
    
    print("[CAM] Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[CAM] Error: cannot open camera"); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    
    worker = InferenceWorker(model, depth_pipe, hazard_model)
    top_alert = {}
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[CAM] Frame read error"); break
            
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            worker.submit(frame)
            res = worker.results
            
            detections = build_detections(res, FRAME_W, FRAME_H)
            
            if detections:
                print(f"[MAIN] Found {len(detections)} detections")
                chosen = pick_top_alert(detections)
                if chosen:
                    msg = build_message(chosen)
                    print(f"[MAIN] Chosen alert: {chosen['type']} - '{msg}'")
                    if speak(msg, chosen["cooldown_key"]):
                        top_alert = chosen
                    else:
                        print(f"[MAIN] speak() failed for: {msg}")
                else:
                    print("[MAIN] All detections in cooldown")
            else:
                top_alert = {}
            
            draw_hud(frame, res, detections, top_alert)
            cv2.imshow("AI Server - Blind Assistance", frame)
            if cv2.waitKey(1) == 27:
                break
    
    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()
        audio_queue.put(None)
        print("[INFO] Stopped.")

if __name__ == "__main__":
    main()
