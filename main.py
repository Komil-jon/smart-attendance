import face_recognition
import numpy as np
import queue
import math
import cv2
import sys
import csv
import os

def face_confidence(face_distance, face_match_threshold=0.6):
    range_val = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range_val * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


class FaceRecognition:
    def __init__(self):
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.cumulative_percentages = {}
        self.shots_count = 0
        self.frame_queue = queue.Queue()
        self.process_this_frame = True

        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image.split('.')[0])
            self.cumulative_percentages[image.split('.')[0]] = 0.0

        print("Encoded faces:", self.known_face_names)

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('Video source not found ...')

        while True:
            ret, frame = video_capture.read()
            if not ret:
                continue

            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Process every other frame
            if self.process_this_frame:
                self.face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=3)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                detected_names = []

                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = 'Unknown'
                    confidence = 'Unknown'

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])
                        detected_names.append(name)
                        print('Detected', name)

                    self.face_names.append(f'{name} ({confidence})')

                # Update the cumulative percentage for each student
                for name in self.known_face_names:
                    if name in detected_names:
                        self.cumulative_percentages[name] += 1.0

                self.shots_count += 1

                # Print the cumulative percentage for each detected person and the shot number
                for name in detected_names:
                    print(f'Shot {self.shots_count}: {name} cumulative percentage: {self.cumulative_percentages[name] / self.shots_count * 100:.2f}%')

            self.process_this_frame = not self.process_this_frame

            # Display annotations
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) == ord('q'):
                print('Finished face recognition')
                break

        attendance_report = self.generate_attendance_report()
        print("Attendance Report:", attendance_report)
        video_capture.release()
        cv2.destroyAllWindows()

    def generate_attendance_report(self):
        attendance_report = []
        for name in self.known_face_names:
            if self.shots_count > 0:
                presence_percentage = (self.cumulative_percentages[name] / self.shots_count) * 100
            else:
                presence_percentage = 0.0
            attendance_report.append([name, f"{presence_percentage:.2f}%"])

        with open("attendance_report.csv", "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Presence Percentage"])
            for row in attendance_report:
                writer.writerow(row)

        return attendance_report


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
