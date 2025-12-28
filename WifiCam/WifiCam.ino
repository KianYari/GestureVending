#include "WifiCam.hpp"
#include <WiFi.h>
#include <WiFiClient.h>
#include <HTTPClient.h>
using namespace esp32cam;
static const char* WIFI_SSID = "Kalle mahi khor";
static const char* WIFI_PASS = "pjtr8569";
static const char* SERVER_IP = "10.143.138.111";
static const int   SERVER_PORT = 5000;

esp32cam::Resolution initialResolution;
WiFiClient client;

void setup() {
  Serial.begin(115200);
  Serial.println();
  
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.println(WiFi.localIP());

  initialResolution = Resolution::find(320, 240);
  // Resolution::find(160, 120)  // QQVGA - ~2-4KB per frame, 40+ FPS
  // Resolution::find(176, 144)  // QCIF  - ~3-5KB per frame, 35+ FPS  
  // Resolution::find(240, 176)  // HQVGA - ~5-8KB per frame, 30+ FPS
  // Resolution::find(320, 240)  // QVGA  - ~8-12KB per frame, 25+ FPS
  
  Config cfg;
  cfg.setPins(pins::AiThinker);
  cfg.setResolution(initialResolution);
  // cfg.setJpeg(80);
  cfg.setJpeg(10);
  // cfg.setJpeg(30);  // Very low quality, max speed
  // cfg.setJpeg(50);  // Low quality, good speed
  // cfg.setJpeg(70);  // Medium quality, decent speed
  cfg.setBufferCount(1);  // Double buffering
  
  if (!Camera.begin(cfg)) {
    Serial.println("Camera init failed");
    ESP.restart();
  }
  
  Serial.println("Camera ready");
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    auto frame = Camera.capture();
    if (!frame) return;

    HTTPClient http;
    String serverUrl = "http://" + String(SERVER_IP) + ":" + String(SERVER_PORT) + "/video_feed";
    
    http.begin(serverUrl);
    http.addHeader("Content-Type", "image/jpeg");
    
    // This sends the data and waits for the server response properly
    int httpResponseCode = http.POST(frame->data(), frame->size());
    
    if (httpResponseCode > 0) {
      // Optional: Read response if you want to use the JSON data
      // String response = http.getString(); 
    } else {
      Serial.printf("Error occurred: %s\n", http.errorToString(httpResponseCode).c_str());
    }
    
    http.end(); // Properly close the connection
  }
  delay(10); // Small delay to prevent CPU saturation
}