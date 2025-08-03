
import cv2
import smtplib
import ssl
from email.message import EmailMessage
from playsound import playsound
import threading
import numpy as np
from twilio.rest import Client
import mysql.connector


thres = 0.55
nmsThres = 0.2
alarm_file = 'alarm.wav'
alarm_playing = False

# Distance estimation
KNOWN_WIDTH = 7.0
FOCAL_LENGTH = 615

# Email setup
EMAIL_SENDER = 'your email id'
EMAIL_PASSWORD = 'generated email password for unique purpose'
EMAIL_RECEIVER = 'email you want to send to'

# Twilio setup
TWILIO_SID = "your id"
TWILIO_AUTH_TOKEN = 'the token generated'
TWILIO_PHONE = 'twilio number'
RECEIVER_PHONE = 'your contact number'

# MySQL setup
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",  # Update if needed
    database="detection_logs"
)
cursor = db.cursor()

# --------------------------- FUNCTIONS --------------------------- #

twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

def send_email_alert(distance):
    try:
        subject = "⚠️ Cell Phone Detected!"
        body = f"A cell phone was detected at {distance:.2f} cm distance."
        em = EmailMessage()
        em['From'] = EMAIL_SENDER
        em['To'] = EMAIL_RECEIVER
        em['Subject'] = subject
        em.set_content(body)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(em)
        print("✅ Email alert sent!")
    except Exception as e:
        print(f"❌ Email error: {e}")

def send_sms_alert(distance):
    try:
        message = twilio_client.messages.create(
            body=f"⚠️ Cell phone detected! Distance: {distance:.2f} cm",
            from_=TWILIO_PHONE,
            to=RECEIVER_PHONE
        )
        print(f"✅ SMS alert sent! SID: {message.sid}")
    except Exception as e:
        print(f"❌ SMS error: {e}")

def play_alarm_sound():
    global alarm_playing
    try:
        alarm_playing = True
        playsound(alarm_file)
    except Exception as e:
        print(f"❌ Alarm sound error: {e}")
    finally:
        alarm_playing = False

def estimate_distance(pixel_width):
    return (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width if pixel_width else 0

def log_to_database(distance):
    try:
        query = "INSERT INTO mobile_detections (distance_cm) VALUES (%s)"
        cursor.execute(query, (distance,))
        db.commit()
        print("📥 Logged to MySQL.")
    except Exception as e:
        print(f"❌ MySQL logging error: {e}")

# - MAIN PROGRAM - #


classNames = []
with open('coco.names', 'rt') as f:
    classNames = f.read().split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Alert control
alert_cooldown = 30
last_alert_time = 0
email_sent = False
sms_sent = False

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Failed to read from webcam.")
        continue

    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)
    cell_phone_detected = False
    current_distance = 0

    try:
        if classIds is not None:
            for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
                if classNames[classId - 1] == 'cell phone':
                    cell_phone_detected = True
                    current_distance = estimate_distance(box[2])

                    cv2.rectangle(img, box, (0, 255, 0), 2)
                    cv2.putText(img,
                                f'Phone {conf * 100:.1f}% | {current_distance:.1f}cm',
                                (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)

                    if not alarm_playing:
                        threading.Thread(target=play_alarm_sound, daemon=True).start()

                    current_time = cv2.getTickCount() / cv2.getTickFrequency()
                    if (current_time - last_alert_time) > alert_cooldown:
                        if not email_sent:
                            threading.Thread(target=send_email_alert, args=(current_distance,), daemon=True).start()
                            email_sent = True
                        if not sms_sent:
                            threading.Thread(target=send_sms_alert, args=(current_distance,), daemon=True).start()
                            sms_sent = True

                        log_to_database(current_distance)
                        last_alert_time = current_time

    except Exception as e:
        print(f"Detection error: {e}")

    if not cell_phone_detected:
        email_sent = False
        sms_sent = False

    cv2.imshow("Cell Phone Detection", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
