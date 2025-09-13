// Coconut Sorting Bench Prototype Firmware
// - Reads HX711 load cell
// - Streams weight (grams) over Serial as: "W: 1035"
// - Listens for commands: R (reject), L (light), M (medium), H (heavy), K (husk)
// - Pulses relays/outputs for actuators

#include "HX711.h"

// --- Pins (adjust to your wiring) ---
const int DOUT = 3;  // HX711 DOUT
const int SCK  = 2;  // HX711 SCK

const int RELAY_REJECT = 5; // Reject diverter
const int RELAY_LIGHT  = 6; // Light bin
const int RELAY_MEDIUM = 7; // Medium bin
const int RELAY_HEAVY  = 8; // Heavy bin
const int RELAY_HUSK   = 9; // Husk motor/solenoid (prototype only)

// --- HX711 ---
HX711 scale;
float calibration_factor = -7050.0; // Calibrate with known weight

// --- Timings ---
const int PULSE_MS = 200;      // Pulse duration for diverter actuations
const int HUSK_MS  = 1000;     // Husk cycle pulse (prototype)

void setup() {
  Serial.begin(115200);

  pinMode(RELAY_REJECT, OUTPUT);
  pinMode(RELAY_LIGHT, OUTPUT);
  pinMode(RELAY_MEDIUM, OUTPUT);
  pinMode(RELAY_HEAVY, OUTPUT);
  pinMode(RELAY_HUSK, OUTPUT);

  digitalWrite(RELAY_REJECT, LOW);
  digitalWrite(RELAY_LIGHT, LOW);
  digitalWrite(RELAY_MEDIUM, LOW);
  digitalWrite(RELAY_HEAVY, LOW);
  digitalWrite(RELAY_HUSK, LOW);

  scale.begin(DOUT, SCK);
  scale.set_scale(calibration_factor);
  scale.tare(); // Reset the scale to 0
}

void loop() {
  // ---- Weight streaming ----
  if (scale.is_ready()) {
    // Average 5 samples; convert kg to grams
    float grams = scale.get_units(5) * 1000.0;
    Serial.print("W: ");
    Serial.println(grams, 0);
  }

  // ---- Command handling ----
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'R') pulse(RELAY_REJECT, PULSE_MS);
    if (c == 'L') pulse(RELAY_LIGHT,  PULSE_MS);
    if (c == 'M') pulse(RELAY_MEDIUM, PULSE_MS);
    if (c == 'H') pulse(RELAY_HEAVY,  PULSE_MS);
    if (c == 'K') pulse(RELAY_HUSK,   HUSK_MS);
  }

  delay(100);
}

void pulse(int pin, int ms) {
  digitalWrite(pin, HIGH);
  delay(ms);
  digitalWrite(pin, LOW);
}