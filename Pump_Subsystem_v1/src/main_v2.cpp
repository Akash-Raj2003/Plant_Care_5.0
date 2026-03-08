#include <Arduino.h>
#include <Servo.h>


Servo myservo;

int liquidLevel = 0; // variable for if there's enough water in tank
int access = 0; // do we have'permission' from machine vision validation to dispense water?
int water_tank_level = 1; // water tank volume in Litres

// variables for water dispensing calc
const int desired_air_rH = 500;
const int water_rH = 0;
const int tubing_volume = 0.000007*1000; //m3
int air_rH = 600;
int current_rH = 350;
int waterRequired = 0;
int adjusted_rH = 0;
int waitTime = 0;
int pumpSpeed = 85/30;

#define PUMPPIN 9 
#define BUTTON 10

class Plant_1{
  public:
      const int pot_volume = 0.5; // in litres
      const int nominal_rH = 250;
      const int nominal_Nc = 125;
};

Plant_1 Test_Plant;

void setup() {
  Serial.begin(9600);
  // water level setup
  pinMode(8, INPUT);
  //pinMode(BUTTON, INPUT);

  // peristaltic pump setup
  myservo.attach(PUMPPIN);

  air_rH = analogRead(A0);
  Serial.println(air_rH);
  delay(10000);

}

void loop() {
  // put your main code here, to run repeatedly:
  liquidLevel = digitalRead(8);
  access = digitalRead(BUTTON);

  adjusted_rH = ((current_rH - Test_Plant.nominal_rH)*(air_rH/desired_air_rH)) + Test_Plant.nominal_rH;

  // water required in L
  waterRequired = (((adjusted_rH - Test_Plant.nominal_rH)/desired_air_rH)*Test_Plant.pot_volume) + tubing_volume;

  //
  waitTime = ((waterRequired*1000)/pumpSpeed)*1000;

  if ((liquidLevel == 1) && (access == 1)){
    myservo.write(0); //clockwise rotation
    Serial.println("PUMPING WATER");
    delay(waitTime);
    access = 0;
    delay(30000);
  }

  else if ((liquidLevel == 1) && (access == 0)){
    myservo.write(90); //stop
    Serial.println("ACCESS REQUIRED");
  }

  else if ((liquidLevel == 0) && (access == 1)){
    myservo.write(90); //stop
    Serial.println("LOW WATER");
  }

  else if ((liquidLevel == 0) && (access == 0)){
    myservo.write(90); //stop
    Serial.println("LOW WATER AND ACCESS REQUIRED");
  }

  else{
    Serial.println("HELP");
  }
  //Serial.println(liquidLevel);
}
