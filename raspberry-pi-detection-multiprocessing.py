from multiprocessing import Process, Manager, cpu_count, set_start_method
import face_recognition
import RPi.GPIO as GPIO
import numpy as np
import threading
import platform
import numpy
import time
import cv2
import os

TOLERANCE = 0.45 # Modify it based on practical results
MELODY_GREEN = [(800, 600)]
MELODY_RED = [(400, 200)]
GPIO.cleanup()
GPIO.setmode(GPIO.BCM)

GPIO.setup(17,GPIO.OUT) # GREEN
GPIO.setup(27,GPIO.OUT) # RED
GPIO.setup(12, GPIO.OUT) # BUZZER
GPIO.setup(14, GPIO.IN)  # PIR sensor input pin (motion sensor)
GPIO.setup(18, GPIO.OUT) # FOR TRIGGER IN SENSOR
GPIO.setup(24, GPIO.IN) # FOR ECHO IN SENSOR\
prev = True

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

def distance():
    global prev
    # set Trigger to HIGH
    GPIO.output(18, True)

    # set Trigger after 0.01ms to LOW
    time.sleep(0.00001)
    GPIO.output(18, False)

    StartTime = time.time()
    StopTime = time.time()

    while GPIO.input(24) == 0:
        StartTime = time.time()

    while GPIO.input(24) == 1:
        StopTime = time.time()

    TimeElapsed = StopTime - StartTime
    distance = (TimeElapsed * 34300) / 2
    print(distance)
    if distance < 50:
        prev = False
        return 777
    else:
        prev = True
        return distance

def next_id(current_id, worker_num):
    if current_id == worker_num:
        return 1
    else:
        return current_id + 1

def prev_id(current_id, worker_num):
    if current_id == 1:
        return worker_num
    else:
        return current_id - 1

def capture(read_frame_list, Global, worker_num):
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    # video_capture.set(3, 640)  # Width of the frames in the video stream.
    # video_capture.set(4, 480)  # Height of the frames in the video stream.
    # video_capture.set(5, 30) # Frame rate.
    print("Width: %d, Height: %d, FPS: %d" % (video_capture.get(3), video_capture.get(4), video_capture.get(5)))

    while not Global.is_exit:
        if Global.buff_num != next_id(Global.read_num, worker_num):
            ret, frame = video_capture.read()
            read_frame_list[Global.buff_num] = frame
            Global.buff_num = next_id(Global.buff_num, worker_num)
        else:
            time.sleep(0.01)

    video_capture.release()

def process(worker_id, read_frame_list, write_frame_list, Global, worker_num):
    known_face_encodings = Global.known_face_encodings
    known_face_names = Global.known_face_names
    while not Global.is_exit:

        while Global.read_num != worker_id or Global.read_num != prev_id(Global.buff_num, worker_num):
            if Global.is_exit:
                break

            time.sleep(0.01)

        if distance() > 50:
            print("No person detected within 50 cm.")
            continue

        time.sleep(Global.frame_delay)
        frame_process = read_frame_list[worker_id]
        Global.read_num = next_id(Global.read_num, worker_num)
        rgb_frame = np.ascontiguousarray(frame_process[:, :, ::-1])

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=TOLERANCE)
            name = "Unknown"
            for i, match in enumerate(matches):
                if match:
                    name = known_face_names[i]
                    break
                    
            cv2.rectangle(frame_process, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame_process, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame_process, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            if name == "Unknown":
                GPIO.output(27, GPIO.HIGH)
                melody(MELODY_RED, 100) # INSTEAD OF DELAY
                GPIO.output(27, GPIO.LOW)
            else:
                GPIO.output(17, GPIO.HIGH)
                melody(MELODY_GREEN, 100) # INSTEAD OF DELAY
                GPIO.output(17, GPIO.LOW)

        while Global.write_num != worker_id:
            time.sleep(0.01)

        write_frame_list[worker_id] = frame_process
        Global.write_num = next_id(Global.write_num, worker_num)

if __name__ == '__main__':

    # Fix Bug on MacOS
    if platform.system() == 'Darwin':
        set_start_method('forkserver')

    # Global variables
    Global = Manager().Namespace()
    Global.buff_num = 1
    Global.read_num = 1
    Global.write_num = 1
    Global.frame_delay = 0
    Global.is_exit = False
    read_frame_list = Manager().dict()
    write_frame_list = Manager().dict()

    if cpu_count() > 2:
        worker_num = cpu_count() - 1 
    else:
        worker_num = 2

    p = []

    p.append(threading.Thread(target=capture, args=(read_frame_list, Global, worker_num,)))
    p[0].start()

    known_face_encodings = []
    known_face_names = []

    with open('users.txt', 'r') as f:
        for line in f:
            name = '_'.join(line.split()[2:])
            image_path = f"photos/{name}.jpg"
            if os.path.exists(image_path):
                print('yedi')
                image = face_recognition.load_image_file(image_path)
                face_encoding = face_recognition.face_encodings(image)[0]
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)
            else:
                print(image_path)

    Global.known_face_encodings = known_face_encodings
    Global.known_face_names = known_face_names

    for worker_id in range(1, worker_num + 1):
        p.append(Process(target=process, args=(worker_id, read_frame_list, write_frame_list, Global, worker_num,)))
        p[worker_id].start()

    last_num = 1
    fps_list = []
    tmp_time = time.time()
    while not Global.is_exit:
        while Global.write_num != last_num:
            last_num = int(Global.write_num)

            # Calculate fps
            delay = time.time() - tmp_time
            tmp_time = time.time()
            fps_list.append(delay)
            if len(fps_list) > 5 * worker_num:
                fps_list.pop(0)
            fps = len(fps_list) / numpy.sum(fps_list)
            print("fps: %.2f" % fps)
            # time.sleep(1) NO NEED FOR SLEEP AS IT IS GETTING GOOD CARE BY melody() AND SLIGHTLY motion()
            
            if fps < 6:
                Global.frame_delay = (1 / fps) * 0.75
            elif fps < 20:
                Global.frame_delay = (1 / fps) * 0.5
            elif fps < 30:
                Global.frame_delay = (1 / fps) * 0.25
            else:
                Global.frame_delay = 0

            # Display the resulting image
            # cv2.imshow('Video', write_frame_list[prev_id(Global.write_num, worker_num)])

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            Global.is_exit = True
            break

        time.sleep(0.01)

    # Quit
    GPIO.cleanup()
    cv2.destroyAllWindows()
