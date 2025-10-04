# 🚙 Raspberry Pi Blind Spot Detection System

## Overview
A real-time blind spot monitoring system for vehicles using Raspberry Pi. It combines four ultrasonic sensors, LEDs, a buzzer, and YOLO-powered camera detection to alert on nearby objects (vehicles or humans) within 1m. The camera activates only on proximity to save resources; the buzzer is toggleable via a button.

## Key Features
- **4-Direction Sensing:**  
  - Left/Right: 1 LED each + buzzer  
  - Front: 2 LEDs + buzzer  
  - Back: shared LED, no buzzer  
- **YOLO Detection:** Configurable for vehicles/humans; default confidence threshold 0.5  
- **Smart Controls:** Camera stops after 3s inactivity; saves annotated frames  
- **CLI Options:** `--conf` (threshold), `--model` (vehicle/human)  

## Hardware
- Raspberry Pi + Camera Module  
- 4x HC-SR04 ultrasonic sensors  
- 5x LEDs  
- Buzzer  
- Button  

## Setup

### GPIO & Connections
| Direction | Trig/Echo | LED(s) |
|-----------|-----------|--------|
| Left      | 2 / 3     | 4      |
| Right     | 14 / 15   | 18     |
| Front     | 20 / 21   | 12, 16 |
| Back      | 1 / 7     | 12     |
| Buzzer    | -         | 17     |
| Button    | -         | 27     |

### Installation
```bash
pip install picamera2 ultralytics opencv-python RPi.GPIO numpy
