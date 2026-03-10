void setup() {
  // put your setup code here, to run once:

  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  Serial.println("001,1");
  delay(2000);
  Serial.println("001,2");
  delay(2000);
}
