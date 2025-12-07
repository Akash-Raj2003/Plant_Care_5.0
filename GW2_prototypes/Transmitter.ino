#include <SPI.h>
#include <RF24.h>
#include <nRF24L01.h>

const int buttonPin = 2;
int buttonState = 0;

RF24 radio(7, 8);  // CE, CSN
const byte address[6] = "00001";

uint8_t Flag = B00000001;  // "moisture detected" 

void setup() {
  Serial.begin(9600);
  pinMode(buttonPin, INPUT);
  while (!Serial) {}
  radio.begin();
  radio.setAutoAck(false);        // turn off ACKs
  radio.setRetries(0, 0);         // no retries
  radio.setDataRate(RF24_1MBPS);  // must match RX
  radio.setChannel(108);          // must match RX
  radio.openWritingPipe(address);
  radio.stopListening();          // TX mode
}

void loop() {
  buttonState = digitalRead(buttonPin);
  //Serial.println(buttonState);

  if (buttonState == HIGH) {
    radio.write(&Flag, sizeof(Flag));  // send value of flag
    Serial.println(Flag); 
    delay(200);  // debounce and avoid switching noise
  }
}
