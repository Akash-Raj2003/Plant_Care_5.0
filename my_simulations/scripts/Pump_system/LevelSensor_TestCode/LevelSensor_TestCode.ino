int liquidLevel = 0;

void setup() {
  Serial.begin(9600);
  pinMode(5, INPUT);
}

void loop() {
  liquidLevel = digitalRead(5);
  Serial.print("liquidLevel = ");
  Serial.println(liquidLevel, DEC);
  delay(500);
}
