import face_recognition
import RPi.GPIO as GPIO
import numpy as np
import time
import cv2
import os

TOLERANCE = 0.45
MELODY_GREEN = [(800, 600)]
MELODY_RED = [(400, 200)]
GPIO.cleanup()
GPIO.setmode(GPIO.BCM)

GPIO.setup(17,GPIO.OUT) # GREEN
GPIO.setup(27,GPIO.OUT) # RED
GPIO.setup(12, GPIO.OUT) # BUZZER
GPIO.setup(14, GPIO.IN)  # PIR sensor input pin (motion sensor)
GPIO.setup(18, GPIO.OUT) # FOR TRIGGER IN SENSOR
GPIO.setup(24, GPIO.IN) # FOR ECHO IN SENSOR

def distance():
    global prev
    GPIO.output(18, True)
    time.sleep(0.00001)
    GPIO.output(18, False)

    StartTime = time.time()
    StopTime = time.time()


    while GPIO.input(24) == 0:
        StartTime = time.time()

    while GPIO.input(24) == 1:
        StopTime = time.time()

    TimeElapsed = StopTime - StartTime
    distance = (TimeElapsed * 34300) / 2#
    print(distance)
    return distance

def melody(melody, pause_duration):
    pwm = GPIO.PWM(12, 440)  # Initialize PWM with a default frequency
    pwm.start(50)  # Start PWM with 50% duty cycle

    try:
        for frequency, duration in melody:
            print(f"Playing Frequency: {frequency} Hz for {duration / 1000.0} seconds")
            pwm.ChangeFrequency(frequency)
            time.sleep(duration / 1000.0)
            pwm.ChangeFrequency(1)  # Set frequency to 1Hz to create a pause without stopping PWM
            time.sleep(pause_duration / 1000.0)
    except KeyboardInterrupt:
        pass
    finally:
        pass
        pwm.stop()
        # GPIO.cleanup() REMOVED DUE TO THE POTENTIAL EFFECT ON THE LED LAMPS
    pwm.stop()

known_face_encodings = []
known_face_names = []

with open('users.txt', 'r') as f:
    for line in f:
        name = '_'.join(line.split()[2:])
        image_path = f"photos/{name}.jpg"
        if os.path.exists(image_path):
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)
        else:
            print(image_path)

face_locations = []
face_encodings = []
face_names = []

def run():
    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()

    if ret:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=TOLERANCE)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            if name == "Unknown":
                GPIO.output(27, GPIO.HIGH)
                melody(MELODY_RED, 100)
                #time.sleep(0.3)
                GPIO.output(27, GPIO.LOW)
            else:
                GPIO.output(17, GPIO.HIGH)
                melody(MELODY_GREEN, 100)
                #time.sleep(0.3)
                GPIO.output(17, GPIO.LOW)
            face_names.append(name)
            print(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    # cv2.imshow('Video', frame)
    video_capture.release()
    # Hit 'q' on the keyboard to quit!
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
    # break

while True:
    if distance() < 50:
        run()
    else:
        continue

# Release handle to the webcam
cv2.destroyAllWindows()
