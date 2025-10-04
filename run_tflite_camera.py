from picamera2 import Picamera2
from ultralytics import YOLO
import RPi.GPIO as GPIO
import cv2
import numpy as np
import os
import time
import argparse
from libcamera import Transform

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Optimized Blind Spot Detection')
parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold (0 to 1)')
parser.add_argument('--model', type=str, default='vehicle', choices=['vehicle', 'human'], help='Model: vehicle or human')
args = parser.parse_args()

# GPIO setup
TRIG_L, ECHO_L, LED_L = 2, 3, 4
TRIG_R, ECHO_R, LED_R = 14, 15, 18
TRIG_F, ECHO_F, LED_F1, LED_F2 = 20, 21, 12, 16
TRIG_B, ECHO_B, LED_B = 1, 7, 12  # LED_B shares with LED_F1
BUZZER, BUTTON = 17, 27

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

for pin in [TRIG_L, TRIG_R, TRIG_F, TRIG_B]:
    GPIO.setup(pin, GPIO.OUT)
for pin in [ECHO_L, ECHO_R, ECHO_F, ECHO_B]:
    GPIO.setup(pin, GPIO.IN)
for pin in [LED_L, LED_R, LED_F1, LED_F2, BUZZER]:
    GPIO.setup(pin, GPIO.OUT)
GPIO.setup(BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Global state
buzzer_enabled = False
last_img_bgr = None  # To persist last processed frame
camera_running = False
last_active_time = time.time() - 4  # Initialize to avoid starting camera immediately if no trigger

# Button callback
def button_callback(channel):
    global buzzer_enabled
    buzzer_enabled = not buzzer_enabled
    print("Buzzer enabled:", buzzer_enabled)

GPIO.add_event_detect(BUTTON, GPIO.RISING, callback=button_callback, bouncetime=300)

# Optimized distance function
def get_distance(TRIG, ECHO):
    GPIO.output(TRIG, False)
    time.sleep(0.0001)
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)
    timeout = time.time() + 0.05  # Increased timeout to 50ms for better reliability
    pulse_start = time.time()
    while GPIO.input(ECHO) == 0 and time.time() < timeout:
        pulse_start = time.time()
    pulse_end = time.time()
    while GPIO.input(ECHO) == 1 and time.time() < timeout:
        pulse_end = time.time()
    if time.time() >= timeout:
        return 999
    duration = pulse_end - pulse_start
    distance = round(duration * 17150, 2)
    print(f"Raw duration: {duration:.6f}s, Calculated distance: {distance}cm")  # Debug print
    return distance

# Verify model file
model_path = f"/home/jack/Downloads/{'human_best.pt' if args.model == 'human' else 'best_v2.pt'}"
if not os.path.exists(model_path):
    print(f"Error: Model file {model_path} not found")
    exit()

# Load YOLO model
try:
    model = YOLO(model_path)
    imgsz = 320  # Fixed for speed
    print(f"Model expects image size: {imgsz}x{imgsz}")
    print(f"Using confidence threshold: {args.conf}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize camera
picam2 = Picamera2()
try:
    config = picam2.create_preview_configuration(
        main={"size": (640, 640), "format": "RGB888"},
        controls={"ExposureTime": 20000, "AnalogueGain": 2.0}
    )
    config["transform"] = Transform(hflip=False, vflip=False)  # Adjust if mirrored
    picam2.configure(config)

    print("System ready for blind spot detection. Press Ctrl+C or 'q' to stop.")

    # Save frames periodically
    os.makedirs('/home/jack/runs/detect', exist_ok=True)
    frame_count = 0
    inference_counter = 0
    start_time = time.time()
    frame_display_count = 0

    while True:
        # Distance measurements
        dist_L = get_distance(TRIG_L, ECHO_L)
        dist_R = get_distance(TRIG_R, ECHO_R)
        dist_F = get_distance(TRIG_F, ECHO_F)
        dist_B = get_distance(TRIG_B, ECHO_B)

        print(f"DL={dist_L}cm | DR={dist_R}cm | DF={dist_F}cm | DB={dist_B}cm | Buzzer={buzzer_enabled}")

        trigger_L = dist_L <= 100
        trigger_R = dist_R <= 100
        trigger_F = dist_F <= 100
        trigger_B = dist_B <= 100
        any_trigger = trigger_L or trigger_R or trigger_F or trigger_B

        num_detections = 0
        buzzer_request = False
        led_l_on = False
        led_r_on = False
        led_f1_on = False
        led_f2_on = False

        if any_trigger:
            last_active_time = time.time()
            if not camera_running:
                print("Object detected within 1m. Starting camera.")
                picam2.start()
                time.sleep(1)  # Warm-up time
                camera_running = True

        if camera_running:
            # Camera and YOLO inference
            frame = picam2.capture_array()
            if inference_counter % 2 == 0:  # Run YOLO every 2nd loop
                if frame is not None:
                    img = frame[:, :, :3]
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    results = model.predict(source=img_bgr, conf=args.conf, imgsz=imgsz, verbose=False)
                    num_detections = len(results[0].boxes)

                    # Draw bounding boxes and labels
                    for box in results[0].boxes:
                        x, y, w, h = map(int, box.xywh[0])
                        cv2.rectangle(img_bgr, (x - w // 2, y - h // 2), 
                                      (x + w // 2, y + h // 2), (0, 255, 0), 2)
                        cls_id = int(box.cls.item())
                        conf_score = box.conf.item()
                        label = f"{'Human' if args.model == 'human' else 'Vehicle'} {cls_id} ({conf_score:.2f})"
                        cv2.putText(img_bgr, label, (x - w // 2, y - h // 2 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(img_bgr, f"{'Humans' if args.model == 'human' else 'Vehicles'}: {num_detections} (conf={args.conf})", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    last_img_bgr = img_bgr  # Persist processed frame

            print(f"{'Humans' if args.model == 'human' else 'Vehicles'} detected: {num_detections}")

            if num_detections > 0:
                last_active_time = time.time()

            # Alert logic
            if num_detections > 0:
                led_l_on = trigger_L
                led_r_on = trigger_R
                led_f1_on = trigger_F or trigger_B
                led_f2_on = trigger_F
                buzzer_request = trigger_L or trigger_R or trigger_F

            GPIO.output(LED_L, led_l_on)
            GPIO.output(LED_R, led_r_on)
            GPIO.output(LED_F1, led_f1_on)
            GPIO.output(LED_F2, led_f2_on)

            if buzzer_enabled and buzzer_request:
                GPIO.output(BUZZER, True)
            else:
                GPIO.output(BUZZER, False)

            # Display last processed frame or current frame
            if last_img_bgr is not None:
                cv2.imshow('Blind Spot Detection', last_img_bgr)
                frame_display_count += 1
            elif frame is not None:
                img_bgr = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)
                cv2.imshow('Blind Spot Detection', img_bgr)
                frame_display_count += 1

            # Calculate and print FPS every 10 frames
            if frame_display_count % 10 == 0 and frame_display_count > 0:
                elapsed_time = time.time() - start_time
                fps = frame_display_count / elapsed_time if elapsed_time > 0 else 0
                print(f"Display FPS: {fps:.2f}")
                start_time = time.time()
                frame_display_count = 0

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if frame_count % 10 == 0 and last_img_bgr is not None:
                cv2.imwrite(f'/home/jack/runs/detect/yolo_frame_{frame_count}.jpg', last_img_bgr)
                print(f"Saved frame to /home/jack/runs/detect/yolo_frame_{frame_count}.jpg")

            inference_counter += 1

            # Check for stopping camera
            if time.time() - last_active_time > 3:
                print("No activity for 3 seconds. Stopping camera.")
                picam2.stop()
                camera_running = False
                last_img_bgr = None
                GPIO.output(LED_L, False)
                GPIO.output(LED_R, False)
                GPIO.output(LED_F1, False)
                GPIO.output(LED_F2, False)
                GPIO.output(BUZZER, False)
                cv2.destroyAllWindows()  # Close window when camera stops

        else:
            # No camera running, ensure alerts are off
            GPIO.output(LED_L, False)
            GPIO.output(LED_R, False)
            GPIO.output(LED_F1, False)
            GPIO.output(LED_F2, False)
            GPIO.output(BUZZER, False)
            time.sleep(0.1)  # Slow down loop when camera is off

        frame_count += 1

except KeyboardInterrupt:
    print("Stopped by User")
except Exception as e:
    print(f"Error: {e}")
finally:
    if camera_running:
        picam2.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()