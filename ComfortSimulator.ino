/*
   Esp32THV.ino - Hybrid HIL Controller for Comfort Control

   This firmware runs on an ESP32 and communicates with a Python hall simulator
   via Serial (115200 baud). It receives simulated environmental values (T, RH, V)
   and a user Comfort‑Control input, maps the three sensor values to a single
   feature using a 3D Hilbert curve (space‑filling curve) to reduce dimensionality,
   then uses a neural network to compute optimal setpoints for temperature,
   relative humidity, and air velocity. The setpoints are sent back to the Python
   simulator, which runs three independent PID controllers.

   Protocol:
     Python -> ESP32: "ENV:<T>,<RH>,<V>"   e.g., "ENV:23.5,55.0,0.25"
     User   -> ESP32: "COMFORT:<value>"    e.g., "COMFORT:75"   (0-100)
     ESP32  -> Python: "SET:<T_set>,<RH_set>,<V_set>"   after each ENV+COMFORT update
*/

#include <Arduino.h>
#include "Esp32THV.h"

// ============================================
//   HARDWARE CONFIGURATION
// ============================================
const int ledPin = 2;          // Built‑in LED for visual feedback
const int serialBaud = 115200; // Must match Python simulator's baud rate

// ============================================
//   PROTOCOL & DATA BUFFERS
// ============================================
String inputString = "";       // Buffer for incoming serial line
bool stringComplete = false;   // Flag when a full line (ending with '\n') is received

// Current simulated environment (received from Python)
float temperature = 25.0;      // °C
float humidity = 50.0;         // %RH
float velocity = 0.25;         // m/s

// User Comfort‑Control (0 = low comfort, 100 = high comfort)
float comfort = 50.0;          // default mid‑range

// Computed setpoints to be sent back to Python
float T_set = 22.0;
float RH_set = 50.0;
float V_set = 0.2;

// ============================================
//   HILBERT CURVE DIMENSIONALITY REDUCTION
// ============================================
const float T_MIN = 0.0;
const float T_MAX = 50.0;
const float RH_MIN = 0.0;
const float RH_MAX = 100.0;
const float V_MIN = 0.0;
const float V_MAX = 1.0;

const uint8_t HILBERT_BITS = 8;
const uint32_t MAX_AXIS = (1UL << HILBERT_BITS) - 1;
const uint32_t MAX_HILBERT_INDEX = (1UL << (3 * HILBERT_BITS)) - 1;

uint32_t hilbert_3d_index(uint32_t x, uint32_t y, uint32_t z, uint8_t bits) {
  const uint8_t n = 3;
  uint32_t X[3] = {x, y, z};
  uint32_t M = 1U << (bits - 1);
  uint32_t Q = M;

  while (Q > 1) {
    uint32_t P = Q - 1;
    for (uint8_t i = 0; i < n; i++) {
      if (X[i] & Q) {
        X[0] ^= P;
      } else {
        uint32_t t = (X[0] ^ X[i]) & P;
        X[0] ^= t;
        X[i] ^= t;
      }
    }
    Q >>= 1;
  }

  for (uint8_t i = 1; i < n; i++) {
    X[i] ^= X[i - 1];
  }
  uint32_t t = 0;
  Q = M;
  while (Q > 1) {
    if (X[n - 1] & Q) {
      t ^= Q - 1;
    }
    Q >>= 1;
  }
  for (uint8_t i = 0; i < n; i++) {
    X[i] ^= t;
  }

  uint32_t h = 0;
  for (int8_t b = bits - 1; b >= 0; b--) {
    for (uint8_t i = 0; i < n; i++) {
      h = (h << 1) | ((X[i] >> b) & 1);
    }
  }
  return h;
}

uint32_t quantize(float val, float min, float max) {
  if (val <= min) return 0;
  if (val >= max) return MAX_AXIS;
  return (uint32_t)((val - min) / (max - min) * MAX_AXIS);
}

float computeHilbertFeature(float t, float rh, float v) {
  uint32_t x = quantize(t, T_MIN, T_MAX);
  uint32_t y = quantize(rh, RH_MIN, RH_MAX);
  uint32_t z = quantize(v, V_MIN, V_MAX);
  uint32_t h = hilbert_3d_index(x, y, z, HILBERT_BITS);
  return (float)h / (float)MAX_HILBERT_INDEX;
}

// ============================================
//   LED CONTROL (Compatible with all ESP32 cores)
// ============================================
void setLedBrightness(uint8_t brightness) {
  // Simple PWM using analogWrite (works on most ESP32 cores)
  // If analogWrite doesn't work, fallback to digitalWrite
  #ifdef ARDUINO_ARCH_ESP32
    // For ESP32, use analogWrite which works on most pins
    analogWrite(ledPin, brightness);
  #else
    if (brightness > 128) {
      digitalWrite(ledPin, HIGH);
    } else {
      digitalWrite(ledPin, LOW);
    }
  #endif
}

// ============================================
//   CONTROL UPDATE
// ============================================
void updateControl() {
  float feature = computeHilbertFeature(temperature, humidity, velocity);
  runNeuralNetwork(comfort, feature, T_set, RH_set, V_set);

  Serial.print("SET:");
  Serial.print(T_set, 1);
  Serial.print(",");
  Serial.print(RH_set, 0);
  Serial.print(",");
  Serial.println(V_set, 2);

  uint8_t brightness = (uint8_t)((T_set - 18.0) / 10.0 * 255);
  if (brightness > 255) brightness = 255;
  setLedBrightness(brightness);
}

// ============================================
//   SERIAL DATA PARSING
// ============================================
void parseCommand(String cmd) {
  if (cmd.startsWith("ENV:")) {
    String params = cmd.substring(4);
    int comma1 = params.indexOf(',');
    int comma2 = params.indexOf(',', comma1 + 1);
    if (comma1 > 0 && comma2 > 0) {
      temperature = params.substring(0, comma1).toFloat();
      humidity = params.substring(comma1 + 1, comma2).toFloat();
      velocity = params.substring(comma2 + 1).toFloat();

      Serial.print("ENV_ACK: T=");
      Serial.print(temperature, 1);
      Serial.print(" RH=");
      Serial.print(humidity, 0);
      Serial.print(" V=");
      Serial.println(velocity, 2);

      updateControl();
    } else {
      Serial.println("ERROR: Invalid ENV format");
    }
  }
  else if (cmd.startsWith("COMFORT:")) {
    float newComfort = cmd.substring(8).toFloat();
    if (newComfort >= 0.0 && newComfort <= 100.0) {
      comfort = newComfort;
      Serial.print("COMFORT_ACK: comfort=");
      Serial.println(comfort, 0);
      updateControl();
    } else {
      Serial.println("ERROR: Comfort must be 0-100");
    }
  }
  else if (cmd == "TEST") {
    Serial.println("ESP32 OK");
  }
  else {
    Serial.println("ERROR: Unknown command");
  }
}

// ============================================
//   SERIAL EVENT HANDLER
// ============================================
void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n') {
      stringComplete = true;
    } else if (inChar != '\r') {
      inputString += inChar;
    }
  }
}

// ============================================
//   NEURAL NETWORK INFERENCE
// ============================================
void runNeuralNetwork(float comfort, float feature,
                      float &t_out, float &rh_out, float &v_out) {
  float input_scaled[2];
  input_scaled[0] = (comfort - pgm_read_float(&INPUT_MEAN[0])) / pgm_read_float(&INPUT_SCALE[0]);
  input_scaled[1] = (feature - pgm_read_float(&INPUT_MEAN[1])) / pgm_read_float(&INPUT_SCALE[1]);

  const int HIDDEN = 16;
  float hidden[HIDDEN];
  for (int i = 0; i < HIDDEN; i++) {
    float sum = pgm_read_float(&HIDDEN_BIAS[i]);
    for (int j = 0; j < 2; j++) {
      sum += input_scaled[j] * pgm_read_float(&INPUT_HIDDEN_WEIGHTS[j * HIDDEN + i]);
    }
    hidden[i] = sum > 0 ? sum : 0;
  }

  float output_scaled[3];
  for (int i = 0; i < 3; i++) {
    float sum = pgm_read_float(&OUTPUT_BIAS[i]);
    for (int j = 0; j < HIDDEN; j++) {
      sum += hidden[j] * pgm_read_float(&HIDDEN_OUTPUT_WEIGHTS[j * 3 + i]);
    }
    output_scaled[i] = sum;
  }

  t_out = output_scaled[0] * pgm_read_float(&OUTPUT_SCALE[0]) + pgm_read_float(&OUTPUT_MEAN[0]);
  rh_out = output_scaled[1] * pgm_read_float(&OUTPUT_SCALE[1]) + pgm_read_float(&OUTPUT_MEAN[1]);
  v_out = output_scaled[2] * pgm_read_float(&OUTPUT_SCALE[2]) + pgm_read_float(&OUTPUT_MEAN[2]);

  t_out = constrain(t_out, 18.0, 28.0);
  rh_out = constrain(rh_out, 30.0, 70.0);
  v_out = constrain(v_out, 0.05, 0.5);
}

// ============================================
//   SETUP
// ============================================
void setup() {
  Serial.begin(serialBaud);
  pinMode(ledPin, OUTPUT);
  
  // Blink LED three times to indicate ready
  for (int i = 0; i < 3; i++) {
    digitalWrite(ledPin, HIGH);
    delay(200);
    digitalWrite(ledPin, LOW);
    delay(200);
  }

  Serial.println("ESP32 THV Controller Ready");
  Serial.println("Protocol: ENV:<T>,<RH>,<V> | COMFORT:<0-100>");
  Serial.println("Sends: SET:<T_set>,<RH_set>,<V_set>");

  updateControl();
}

// ============================================
//   MAIN LOOP
// ============================================
void loop() {
  if (stringComplete) {
    if (inputString.length() > 0) {
      parseCommand(inputString);
    }
    inputString = "";
    stringComplete = false;
  }
}