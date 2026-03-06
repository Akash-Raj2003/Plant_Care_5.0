#include <SPI.h>
#include <RH_RF95.h>


// Lora pin definitions for cd, rst, and interrupt pin
#define RFM95_CS   10   
#define RFM95_RST  9
#define RFM95_INT  2    

#define RF95_FREQ  433.0 //set frequency to 433 MHz

RH_RF95 rf95(RFM95_CS, RFM95_INT);

int packets[100] = {0};
int packetNum = 0;

void resetRadio() { //force a hard reset of sensor at start of code
  pinMode(RFM95_RST, OUTPUT);
  digitalWrite(RFM95_RST, HIGH); delay(10);
  digitalWrite(RFM95_RST, LOW);  delay(10);
  digitalWrite(RFM95_RST, HIGH); delay(10);
}

void setup() {
  Serial.begin(9600);
  delay(200);


// Lora setup
/////////////////////////////////////////////////////////////////
  resetRadio();

  if (!rf95.init()) { //initialise Loras
    Serial.println("LoRa failed");
  }

  if (!rf95.setFrequency(RF95_FREQ)) {
    Serial.println("Frequency set failed!");
    while (1);
  }
  rf95.setTxPower(10, false); //set power to 10dB so it doesnt use too much current

  Serial.println("Transmitter ready");
}

void loop() {
  if (rf95.available()) { //if Loras are available, receive radio packet and send value to python GUI serial
    char payload[32] = {0};
    uint8_t len = sizeof(payload);
    
    if(packetNum >= 0 && packetNum <= 100){
      if (rf95.recv((uint8_t*)payload, &len)) {
        Serial.println(payload);  // e.g. 0001,2
        packetNum = packetNum + 1;
        packets[packetNum] = packetNum;

      }
    }
    
}
}
  

