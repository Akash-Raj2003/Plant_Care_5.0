#include <Servo.h>

Servo myservo;

#define PUMPPIN 8

// ----------------------
// Pump parameters
// ----------------------
float pumpSpeed = 32.0 / 60.0;   // mL per second
float tubing_volume = 0.000007 * 1000; // L already in tubing

// ----------------------
// Soil / plant parameters
// ----------------------
float V_soil = 0.5;     // soil water volume interacting with fertiliser (L)

// Nitrogen concentrations (mg/L)
float C_target = 125.0;   // desired nitrogen concentration
float C_soil = 120.0;      // measured soil nitrogen concentration

// ----------------------
// Fertiliser parameters
// ----------------------
float C_fert = 3500.0;    // fertiliser nitrogen concentration (mg/L)

// Dilution ratio
float V_f = 0.1;          // fertiliser used in mix (L)
float V_w = 1.0;          // water used in mix (L)

// ----------------------
float fertRequired = 0;
float waitTime = 0;

void setup() {

  Serial.begin(9600);

  myservo.attach(PUMPPIN);
  myservo.write(90); // ensure pump starts OFF
}

void loop() {

  // -----------------------------
  // Check if fertiliser is needed
  // -----------------------------
  if (C_target <= C_soil) {
    Serial.println("Nitrogen level sufficient - no fertiliser required");
  }
  else {
    // --------------------------------------
    // Fertiliser dosing calculation
    // --------------------------------------
    fertRequired =
      ((C_target - C_soil) * V_soil * (V_f + V_w)) /
      (C_fert * V_f);

    // add tubing volume compensation
    fertRequired = fertRequired;

    Serial.print("Fertiliser Required (L): ");
    Serial.println(fertRequired, 6);

    // Convert to mL
    float fert_mL = fertRequired * 1000;

    Serial.print("Fertiliser Required (mL): ");
    Serial.println(fert_mL);

    // Pump runtime calculation
    waitTime = (fert_mL / pumpSpeed) * 1000;

    Serial.print("Pump Runtime (ms): ");
    Serial.println(waitTime);

    // -----------------------------
    // Run pump
    // -----------------------------
    myservo.write(0);   // start pump
    delay((unsigned long)waitTime);
    myservo.write(90);  // stop pump
  }
  delay(30000);  // run loop every 30 seconds
}