///https://learn.sparkfun.com/tutorials/flex-sensor-hookup-guide/all#example-circuit
//https://forum.arduino.cc/index.php?topic=89941.0
/******************************************************************************
Flex_Sensor_Example.ino
Example sketch for SparkFun's flex sensors
  (https://www.sparkfun.com/products/10264)
Jim Lindblom @ SparkFun Electronics
April 28, 2016

Create a voltage divider circuit combining a flex sensor with a 47k resistor.
- The resistor should connect from A0 to GND.
- The flex sensor should connect from A0 to 3.3V
As the resistance of the flex sensor increases (meaning it's being bent), the
voltage at A0 should decrease.

Development environment specifics:
Arduino 1.6.7
******************************************************************************/


/******************************************************************************
 * 
 * For glove project:
 * resistor for 3.3" sensor [0 degree 33.5K, 90 degree 90K] Circuit use R =43K
 * resistor for 5.4" sensor [0 degree 8.5K, 90 degree 13K] Circuit use R =10K
 */

const int FLEX_PIN_1 = A0;  //3 thumb        // Pin connected to voltage divider output
const int FLEX_PIN_2 = A1;  //3 pinky
const int FLEX_PIN_3 = A2;  //3 ring
const int FLEX_PIN_4 = A3;  //3 middle
const int FLEX_PIN_5 = A4;  //3 index
const int FLEX_PIN_6 = A5;  //2 thumb
const int FLEX_PIN_7 = A8;  //2 pinky
const int FLEX_PIN_8 = A9;  //2 ring
const int FLEX_PIN_9 = A10;  //2 middle
const int FLEX_PIN_10 = A11;  //2 index
const int FLEX_PIN[] = {FLEX_PIN_6, FLEX_PIN_5, FLEX_PIN_4, FLEX_PIN_3, FLEX_PIN_2, FLEX_PIN_1,FLEX_PIN_10,FLEX_PIN_9,FLEX_PIN_8,FLEX_PIN_7}; //Long sensor, short sensor. From pinky to thumb?
                     //{2 Thumb,    3 Index,    3 Middle,   3 Ring,     3 Pinky,     1 Thumb,   2 index,   2 middle,   2 ring,    2 pinky}

// Measure the voltage at 5V and the actual resistance of your
// 47k resistor, and enter them below:
const float VCC = 4.98; // Measured voltage of Ardunio 5V line/// could be replaced by external power with 4.98V

const float R_DIV_1 = 43000.0; // Measured resistance of 3.3" f sensor
const float R_DIV_2 = 10000.0; // Measured resistance of 5.5" f sensor ** resistor does not affect much of the result


// Upload the code, then try to adjust these values to more
// accurately calculate bend degree.
/* STRAIGHT_RESISTANCE = SR
 * BEND_RESISTANCE = BR
 */
//const float STRAIGHT_RESISTANCE = 8500.0; // resistance when straight [5.5" flex sensor]
//const float BEND_RESISTANCE = 13000.0; // resistance at 90 deg [5.5" flex sensor]
//const float STRAIGHT_RESISTANCE = 335000.0; // resistance when straight [3.3" flex sensor]
//const float BEND_RESISTANCE = 90000.0; // resistance at 90 deg [3.3" flex sensor]

void setup() 
{
  Serial.begin(9600);
  for (int i=0; i<10; i++)
  {
    pinMode(FLEX_PIN[i], INPUT);
  }
}

void loop() 
{
  int val = Serial.read()- '0';

    // 1.Read the ADC, and calculate voltage and resistance from it
    int flexADC[] = {0,0,0,0,0,0,0,0,0,0};
    float angle[] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};

    //int ADCs = FLEX_PIN; using for loop:
    for (int i=0; i<10; i++)
      {
        flexADC[i] = analogRead(FLEX_PIN[i]);
      }
  
    // 2. set and adjust the angle setting by the reading results: angle[i] = map(flexADC[i], 'flat ADC', '180//90 degree ADC', 0, 180.0);
    // need to calibrate every time before testing  
      angle[0] = map(flexADC[0], 813, 767, 0, 90.0); //821, 792
      angle[1] = map(flexADC[1], 483, 417, 0, 90.0);
      angle[2] = map(flexADC[2], 457, 392, 0, 90.0);
      angle[3] = map(flexADC[3], 463, 369, 0, 90.0);
      angle[4] = map(flexADC[4], 457, 393, 0, 90.0);
      //cross first joint
      angle[5] = map(flexADC[5], 437, 428, 0, 90.0); //506, 466
      angle[6] = map(flexADC[6], 511, 558, 0, 90.0);
      angle[7] = map(flexADC[7], 556, 601, 0, 90.0);
      angle[8] = map(flexADC[8], 534, 581, 0, 90.0);
      angle[9] = map(flexADC[9], 492, 545, 0, 90.0);
  
    //Serial.println("Resistance: " + String(flexR) + " ohms");
    //Serial.println("Resistance: " + String(flexR) + " ohms");
    // Use the calculated resistance to estimate the sensor's
    // bend angle:
      //  //3. testing the bend angle:
        //Serial.println(String(flexADC[0]));
        //Serial.println(String(flexADC[1]));
        //Serial.println(String(flexADC[2]));
        //Serial.println(String(flexADC[3]));
        //Serial.println(String(flexADC[4]));

        //Serial.println(String("."));
        
        //Serial.println(String(flexADC[5]));
        //Serial.println(String(flexADC[6]));
        //Serial.println(String(flexADC[7]));
        //Serial.println(String(flexADC[8]));
        //Serial.println(String(flexADC[9]));
      
      ////  
      //  Serial.println("Bend: " + String(angle[0]) + " degrees");
      //  Serial.println("Bend: " + String(angle[1]) + " degrees");
      //  Serial.println("Bend: " + String(angle[2]) + " degrees");
      //  Serial.println("Bend: " + String(angle[3]) + " degrees");
      //  Serial.println("Bend: " + String(angle[4]) + " degrees");
      //  Serial.println("Bend: " + String(angle[5]) + " degrees");
      //  Serial.println();
      //

      //  
      //  Serial.println("Bend: " + String(angle[6]) + " degrees");
      //  Serial.println("Bend: " + String(angle[7]) + " degrees");
      //  Serial.println("Bend: " + String(angle[8]) + " degrees");
      //  Serial.println("Bend: " + String(angle[9]) + " degrees");
      //  Serial.println();

    //4. recording the data:
    //Thumb[1]//Index//Middle//Ring//Little//Thumb[2]
    Serial.println(String(angle[0]));
    Serial.println(String(angle[1]));
    Serial.println(String(angle[2]));
    Serial.println(String(angle[3]));
    Serial.println(String(angle[4]));
  
    Serial.println(String(angle[5]));
    Serial.println(String(angle[6]));
    Serial.println(String(angle[7]));
    Serial.println(String(angle[8]));
    Serial.println(String(angle[9]));
//    Serial.println(String("___________"));
  
  delay(100);
}
