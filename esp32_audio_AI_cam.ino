/**
 * ESP32 - AI Cam Audio + Sensor Backup
 * ESP32 = TCP CLIENT, connect den Python server
 * Python gui PCM → ESP32 nhan → phat I2S
 *
 * PINOUT:
 *  I2S MAX98357A : BCK=26, WS=25, DIN=22
 *  HC-SR04 #1    : TRIG=32, ECHO=34 (+1k ohm)
 *  HC-SR04 #2    : TRIG=18, ECHO=35 (+1k ohm)
 *  Buzzer        : GPIO 4
 *  Motor rung    : GPIO 5 (qua NPN transistor)
 *  LED           : GPIO 2
 */

#include <WiFi.h>
#include <driver/i2s.h>

const char* ssid        = "iPhone";
const char* password    = "123456789";
const char* PYTHON_IP   = "172.20.10.2";   // IP may tinh chay ai_server.py
const int   PYTHON_PORT = 5000;

// Không dùng static IP cho iPhone hotspot - để DHCP tự động
// IPAddress local_IP(192, 168, 1, 100);
// IPAddress gateway(192, 168, 1, 1);
// IPAddress subnet(255, 255, 255, 0);
// IPAddress dns(8, 8, 8, 8);

// I2S
#define I2S_BCK         26
#define I2S_WS          25
#define I2S_DOUT        22
#define DMA_BUF_COUNT   16
#define DMA_BUF_LEN     512
#define DMA_TOTAL_BYTES (DMA_BUF_COUNT * DMA_BUF_LEN * 2)

// Sensor
#define TRIG_OBSTACLE   32
#define ECHO_OBSTACLE   34
#define TRIG_HOLE       18
#define ECHO_HOLE       35

// Output
#define BUZZER_PIN      4
#define VIBRO_PIN       5
#define LED_PIN         2

// Nguong (cm)
#define OBSTACLE_DANGER   40
#define OBSTACLE_WARN     80
#define HOLE_BASELINE     40
#define HOLE_THRESHOLD    20
#define HOLE_STEP          5

// Cooldown (ms)
#define CD_OBSTACLE_DANGER  2000
#define CD_OBSTACLE_WARN    4000
#define CD_HOLE             3000
#define CD_EMERGENCY        1000

volatile bool serverOnline     = false;
unsigned long lastObstacleBuzz = 0;
unsigned long lastHoleBuzz     = 0;

void i2s_init() {
  i2s_config_t cfg = {
    .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
    .sample_rate          = 16000,
    .bits_per_sample      = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format       = I2S_CHANNEL_FMT_ONLY_RIGHT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags     = 0,
    .dma_buf_count        = DMA_BUF_COUNT,
    .dma_buf_len          = DMA_BUF_LEN,
    .use_apll             = true
  };
  i2s_pin_config_t pins = {
    .bck_io_num   = I2S_BCK,
    .ws_io_num    = I2S_WS,
    .data_out_num = I2S_DOUT,
    .data_in_num  = I2S_PIN_NO_CHANGE
  };
  i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pins);
}

void i2s_flush_silence() {
  static uint8_t silence[DMA_TOTAL_BYTES];
  memset(silence, 0, sizeof(silence));
  size_t written;
  i2s_write(I2S_NUM_0, silence, sizeof(silence), &written, pdMS_TO_TICKS(500));
  i2s_zero_dma_buffer(I2S_NUM_0);
}

// Audio Task (Core 0): connect Python, nhan PCM, phat I2S
void audioTask(void* param) {
  WiFiClient client;
  uint8_t buf[2048];
  unsigned long lastDataTime = 0;
  const unsigned long KEEPALIVE_TIMEOUT = 30000; // 30s không có data thì reconnect

  for (;;) {
    Serial.printf("[Audio] Connecting to Python %s:%d\n", PYTHON_IP, PYTHON_PORT);
    if (!client.connect(PYTHON_IP, PYTHON_PORT)) {
      Serial.println("[Audio] Connect failed, retry in 5s");
      serverOnline = false;
      digitalWrite(LED_PIN, LOW);
      vTaskDelay(pdMS_TO_TICKS(5000));
      continue;
    }

    client.setNoDelay(true);
    client.setTimeout(100); // timeout ngắn để không block
    serverOnline = true;
    digitalWrite(LED_PIN, HIGH);
    lastDataTime = millis();
    Serial.println("[Audio] Connected to Python server");

    // Nhan PCM tu Python va phat I2S
    unsigned long lastDebug = 0;
    unsigned long bytesReceived = 0;
    unsigned long packetsReceived = 0;
    
    while (client.connected()) {
      int avail = client.available();
      if (avail > 0) {
        int len = client.read(buf, min(avail, (int)sizeof(buf)));
        if (len > 0) {
          lastDataTime = millis();
          bytesReceived += len;
          packetsReceived++;
          
          // BỎ gain trên ESP32 - chỉ dùng gain từ Python
          // Gain trên cả 2 bên gây distortion
          
          size_t written;
          i2s_write(I2S_NUM_0, buf, len, &written, portMAX_DELAY);
          
          // Debug stats mỗi 5 giây
          if (millis() - lastDebug > 5000) {
            Serial.printf("[Audio] Stats: %lu bytes, %lu packets, avg: %lu bytes/pkt\n",
                         bytesReceived, packetsReceived, 
                         packetsReceived > 0 ? bytesReceived/packetsReceived : 0);
            lastDebug = millis();
          }
        }
      } else {
        // Kiểm tra timeout - nếu quá lâu không có data, giữ kết nối
        if (millis() - lastDataTime > KEEPALIVE_TIMEOUT) {
          Serial.println("[Audio] Keepalive timeout, reconnecting...");
          break;
        }
        vTaskDelay(pdMS_TO_TICKS(50)); // giảm CPU usage
      }
    }

    client.stop();
    serverOnline = false;
    digitalWrite(LED_PIN, LOW);
    i2s_flush_silence();
    Serial.println("[Audio] Disconnected, retry in 5s");
    vTaskDelay(pdMS_TO_TICKS(5000));
  }
}

// Sensor
float measureCm(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH); delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  long dur = pulseIn(echoPin, HIGH, 30000);
  if (dur == 0) return 999.0;
  return dur * 0.0343 / 2.0;
}

void buzzPattern(int times, int onMs, int offMs) {
  for (int i = 0; i < times; i++) {
    digitalWrite(BUZZER_PIN, HIGH); delay(onMs);
    digitalWrite(BUZZER_PIN, LOW);
    if (i < times - 1) delay(offMs);
  }
}

void vibratePattern(int times, int onMs, int offMs) {
  for (int i = 0; i < times; i++) {
    digitalWrite(VIBRO_PIN, HIGH); delay(onMs);
    digitalWrite(VIBRO_PIN, LOW);
    if (i < times - 1) delay(offMs);
  }
}

void alert(int times, int onMs, int offMs) {
  buzzPattern(times, onMs, offMs);
  vibratePattern(times, onMs, offMs);
}

void setup() {
  Serial.begin(115200);
  delay(1000);  // Đợi serial port ổn định
  Serial.println("\n\n=== ESP32 AI CAM STARTING ===");
  
  // In thông tin hệ thống
  Serial.printf("Free heap: %d bytes\n", ESP.getFreeHeap());
  Serial.printf("Chip model: %s\n", ESP.getChipModel());
  Serial.printf("CPU freq: %d MHz\n", ESP.getCpuFreqMHz());
  
  pinMode(TRIG_OBSTACLE, OUTPUT);
  pinMode(ECHO_OBSTACLE, INPUT);
  pinMode(TRIG_HOLE,     OUTPUT);
  pinMode(ECHO_HOLE,     INPUT);
  pinMode(BUZZER_PIN,    OUTPUT);
  pinMode(VIBRO_PIN,     OUTPUT);
  pinMode(LED_PIN,       OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(VIBRO_PIN,  LOW);
  digitalWrite(LED_PIN,    LOW);

  i2s_init();

  // Không dùng WiFi.config() - để DHCP tự động cấp IP
  // WiFi.config(local_IP, gateway, subnet, dns);
  WiFi.begin(ssid, password);
  Serial.print("WiFi connecting");
  while (WiFi.status() != WL_CONNECTED) {
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    delay(500);
    Serial.print(".");
  }
  digitalWrite(LED_PIN, HIGH);
  Serial.println("\nWiFi OK: " + WiFi.localIP().toString());

  // Test buzzer và vibration
  Serial.println("Testing buzzer and vibration...");
  alert(2, 100, 100);
  delay(500);
  
  // Test sensor
  Serial.println("Testing sensors...");
  float testObs = measureCm(TRIG_OBSTACLE, ECHO_OBSTACLE);
  float testHole = measureCm(TRIG_HOLE, ECHO_HOLE);
  Serial.printf("Sensor test: Obstacle=%.1fcm, Hole=%.1fcm\n", testObs, testHole);

  xTaskCreatePinnedToCore(audioTask, "AudioTask", 4096, NULL, 2, NULL, 0);
  
  Serial.println("=== SETUP COMPLETE ===");
  Serial.println("Backup system will activate if server is offline");
}

void loop() {
  unsigned long now = millis();
  float distObstacle = measureCm(TRIG_OBSTACLE, ECHO_OBSTACLE);
  float distHole     = measureCm(TRIG_HOLE,     ECHO_HOLE);
  
  // Debug: in ra giá trị sensor + memory
  static unsigned long lastMemCheck = 0;
  if (now - lastMemCheck > 10000) {  // Mỗi 10 giây
    Serial.printf("[SYSTEM] Free heap: %d bytes, Server: %s\n", 
                  ESP.getFreeHeap(), serverOnline ? "ON" : "OFF");
    lastMemCheck = now;
  }
  
  Serial.printf("[SENSOR] Obs=%.1f Hole=%.1f Server=%s\n",
                distObstacle, distHole, serverOnline ? "ON" : "OFF");

  bool holeDetected = (distHole > HOLE_BASELINE + HOLE_THRESHOLD);
  bool stepDetected = (distHole < HOLE_STEP);

  // BACKUP SYSTEM: hoạt động khi server OFFLINE
  if (!serverOnline) {
    Serial.println("[BACKUP] Server offline - backup system active");
    
    // Kiểm tra vật cản
    if (distObstacle < OBSTACLE_DANGER && distObstacle > 0) {
      if (now - lastObstacleBuzz > CD_OBSTACLE_DANGER) {
        Serial.println("[BACKUP] DANGER! Obstacle detected");
        lastObstacleBuzz = now;
        alert(3, 80, 80);
      }
    } else if (distObstacle < OBSTACLE_WARN && distObstacle > 0) {
      if (now - lastObstacleBuzz > CD_OBSTACLE_WARN) {
        Serial.println("[BACKUP] WARNING! Obstacle detected");
        lastObstacleBuzz = now;
        alert(1, 200, 0);
      }
    }
    
    // Kiểm tra hố/bậc thang
    if (holeDetected && now - lastHoleBuzz > CD_HOLE) {
      Serial.println("[BACKUP] HOLE detected!");
      lastHoleBuzz = now;
      buzzPattern(1, 500, 0);
      vibratePattern(2, 300, 150);
    } else if (stepDetected && now - lastHoleBuzz > CD_HOLE) {
      Serial.println("[BACKUP] STEP detected!");
      lastHoleBuzz = now;
      alert(2, 150, 100);
    }
  } else {
    // Server online: chỉ cảnh báo khẩn cấp
    if (distObstacle < 20 && distObstacle > 0 && now - lastObstacleBuzz > CD_EMERGENCY) {
      Serial.println("[EMERGENCY] Very close obstacle!");
      lastObstacleBuzz = now;
      alert(5, 50, 50);
    }
  }
  
  delay(100);
}
