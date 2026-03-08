// #include <Arduino.h>
// #include <Servo.h>

// Servo myservo;

// int liquidLevel = 0;
// int access = 0;

// #define PUMPPIN 9 
// #define BUTTON 10

// void setup() {
//   Serial.begin(9600);
//   // water level setup
//   pinMode(8, INPUT);
//   //pinMode(BUTTON, INPUT);

//   // peristaltic pump setup
//   myservo.attach(PUMPPIN);
  
// }

// void loop() {
//   // put your main code here, to run repeatedly:
//   liquidLevel = digitalRead(8);
//   access = digitalRead(BUTTON);

//   if ((liquidLevel == 1) && (access == 1)){
//     myservo.write(0); //clockwise rotation
//     Serial.println("PUMPING WATER");
//   }

//   else if ((liquidLevel == 1) && (access == 0)){
//     myservo.write(90); //stop
//     Serial.println("ACCESS REQUIRED");
//   }

//   else if ((liquidLevel == 0) && (access == 1)){
//     myservo.write(0); //stop
//     Serial.println("pumping water");
//   }

//   else if ((liquidLevel == 0) && (access == 0)){
//     myservo.write(90); //stop
//     Serial.println("LOW WATER AND ACCESS REQUIRED");
//   }

//   else{
//     Serial.println("HELP");
//   }
//   //Serial.println(liquidLevel);
// }
