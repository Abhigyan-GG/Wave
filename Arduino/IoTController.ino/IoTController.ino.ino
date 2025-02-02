#include <WiFi.h>
#include <ArduinoWebsockets.h>
#include <ArduinoJson.h>

// WiFi credentials
const char* ssid = "Airtel_hema_9037";
const char* password = "Air30338";

// WebSocket server URL
const char* websockets_server_url = "ws://192.168.1.4:8765";

// Pin definitions
const int fanPin = 5; // Example GPIO for fan control
const int meterPin = 18; // Example GPIO for meter control (PWM)

using namespace websockets;

WebsocketsClient client;

void onMessageCallback(WebsocketsMessage message) {
    Serial.println("Received message: " + message.data());

    // Parse JSON
    StaticJsonDocument<200> doc;
    DeserializationError error = deserializeJson(doc, message.data());

    if (error) {
        Serial.print(F("deserializeJson() failed: "));
        Serial.println(error.f_str());
        return;
    }

    const char* type = doc["type"];

    if (strcmp(type, "state") == 0) {
        bool fanState = doc["fan_state"];
        int meterValue = doc["meter_value"];

        // Control fan
        digitalWrite(fanPin, fanState ? HIGH : LOW);

        // Control meter (e.g., LED brightness or motor speed)
        analogWrite(meterPin, meterValue);

        Serial.print("Fan State: ");
        Serial.println(fanState);
        Serial.print("Meter Value: ");
        Serial.println(meterValue);
    }
}

void setup() {
    Serial.begin(115200);
    pinMode(fanPin, OUTPUT);
    pinMode(meterPin, OUTPUT);

    // Connect to WiFi
    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nConnected to WiFi");

    // Connect to WebSocket server
    client.onMessage(onMessageCallback);
    client.connect(websockets_server_url);

    Serial.println("Connected to WebSocket server");
}

void loop() {
    client.poll();
}
