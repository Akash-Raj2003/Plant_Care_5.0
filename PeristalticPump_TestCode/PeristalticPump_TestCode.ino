#include <Servo.h>

Servo myservo;

#define PUMPPIN 9 //pump control pin
#define waitTime 10000 //interval time (ms)

void setup() {
  myservo.attach(PUMPPIN); //'connect' servo control to output pin 9
}

void loop() {
  myservo.write(0);  //clockwise max speed
  // delay(waitTime);
  // myservo.write(90);  //stop
  // delay(waitTime);
  // myservo.write(180);  //anticlockwise max speed
  // delay(waitTime);
  // myservo.write(90);  //stop
  // delay(waitTime);

}
