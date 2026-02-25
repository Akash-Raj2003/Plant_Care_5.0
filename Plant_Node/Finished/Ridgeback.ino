#include <SPI.h>
#include <RF24.h>
#include <nRF24L01.h>

RF24 radio(7, 8);                 // CE, CSN pins
const byte address[6] = "00001"; //adress of NRF

void setup() {
  Serial.begin(9600);
  radio.begin(); //sets up radio
  radio.setChannel(76);
  radio.setDataRate(RF24_1MBPS);
  radio.setPALevel(RF24_PA_LOW);   // start LOW or MIN for testing
  radio.openReadingPipe(0, address);
  radio.startListening();

  Serial.println("Receiver ready");
}

void loop() {
  if (radio.available()) {      //if radio message received, output message into serial for node communication in ROS          
    char payload[32] = {0};
    radio.read(payload, sizeof(payload));
    Serial.println(payload);
  }
}