#include <SPI.h>
#include <nRF24L01.h>
#include <RF24.h>

RF24 radio(7, 8);  // CE, CSN

const byte address[6] = "00001";

const int LED_PIN = 3;  //blue LED

uint8_t Flag = 0;

void setup() {
  Serial.begin(9600);
  while (!Serial) {}
  radio.begin();
  radio.setAutoAck(false);        
  radio.setDataRate(RF24_1MBPS);  // must match TX
  radio.setChannel(108);          // must match TX
  radio.openReadingPipe(0, address); //same address
  radio.startListening(); //start listening
}

void loop() {

  //////////////////////////////////////////////////////////
  //ROS UR5 side
  /////////////////////////////////////////////////////////
  if (radio.available()) { //only operates if found a radio signal

    radio.read(&Flag, sizeof(Flag)); //overwrites value in Flag
    if(Flag == B0000001){
      Serial.println("water");
      Flag = B00000000; //sets back to zero
    }
  }

////////////////////////////////////////////////////////////////////
// Water pump side
////////////////////////////////////////////////////////////////////
  if (Serial.available()) { //monitors the RX from the usb so will not operate when printing to serial TX
    String message = Serial.readStringUntil('\n');  // read full message

    message.trim();  // remove whitespace

    if (message == "Done") {
      digitalWrite(LED_PIN, HIGH);   // turn blue LED on
      Serial.println("Watering");
    }

  }
  }
