#include <SPI.h>
#include <RF24.h>
#include <nRF24L01.h>

RF24 radio(7, 8);                 // CE, CSN
const byte address[6] = "00001";  // must match transmitter

void setup() {
  Serial.begin(9600);

  radio.begin();
  radio.openReadingPipe(0, address);
  /*
  radio.openReadingPipe(0, "NODE1"); like this for other nodes
  radio.openReadingPipe(1, "NODE2");
  radio.openReadingPipe(2, "NODE3");
*/
  radio.setPALevel(RF24_PA_LOW);
  radio.startListening();

  Serial.println("Receiver ready");
}

void loop() {
  if (radio.available()) {
    char payload[32] = {0};              // clear buffer every time
    radio.read(payload, sizeof(payload)); // read max 32 bytes

    Serial.println(payload);             // prints e.g. 0001,2
  }
}
