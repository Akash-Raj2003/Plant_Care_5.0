#include <Servo.h>

Servo myservo;

float desired_air_rH = 500.0;
float water_rH = 0.0;
float tubing_volume = 0.000007 * 1000;

float air_rH = 400.0;  /// change for testing
float current_rH = 200.0;   /// change for testing

float waterRequired = 0;
float adjusted_rH = 0;
float waitTime = 0;
float pumpSpeed = 32.0 / 60.0;

float pot_volume = 0.5;
float nominal_rH = 250.0;

#define PUMPPIN 8

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600); // peristaltic pump setup 
  myservo.attach(PUMPPIN);
}

void loop() {
  // put your main code here, to run repeatedly:

  adjusted_rH = ((current_rH - nominal_rH) * (air_rH / desired_air_rH)) + nominal_rH;

  Serial.print("Adjusted RH: ");
  Serial.println(adjusted_rH);

  waterRequired = (((adjusted_rH - nominal_rH) / desired_air_rH) * pot_volume) + tubing_volume;

  if (waterRequired <= 0) {
    Serial.println("No water required");
  }
  else {

    Serial.print("Water Required: ");
    Serial.println(waterRequired, 6);

    waitTime = ((waterRequired * 1000) / pumpSpeed) * 1000;

    Serial.print("WaitTime: ");
    Serial.println(waitTime);

    myservo.write(0);

    delay((unsigned long)waitTime);

    myservo.write(90); // stop pump
  }
  delay(30000);
}
}
