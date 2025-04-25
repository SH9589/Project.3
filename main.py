import cv2
import numpy as np
import speech_recognition as sr
import pyttsx3
import threading
import time
from pathlib import Path
import face_recognition
from ultralytics import YOLO

class Jarvis:
    def __init__(self):
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Initialize text-to-speech
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # Initialize camera
        self.camera = cv2.VideoCapture(0)
        self.show_camera = False
        self.camera_thread = None
        
        # Initialize face recognition
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
        
        # Initialize object detection
        self.model = YOLO('yolov8n.pt')
        
        # Initialize NLP
        self.commands = {
            "show camera": self.show_camera_feed,
            "hide camera": self.hide_camera_feed,
            "who do you see": self.identify_faces,
            "what objects do you see": self.detect_objects,
            "exit": self.exit_program
        }
        
    def load_known_faces(self):
        # Load known faces from data directory
        data_dir = Path("data/faces")
        if not data_dir.exists():
            data_dir.mkdir(parents=True)
            return
            
        for face_file in data_dir.glob("*.jpg"):
            image = face_recognition.load_image_file(face_file)
            face_encoding = face_recognition.face_encodings(image)[0]
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(face_file.stem)
            
    def speak(self, text):
        print(f"Jarvis: {text}")
        self.engine.say(text)
        self.engine.runAndWait()
        
    def listen(self):
        with self.microphone as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)
            
        try:
            command = self.recognizer.recognize_google(audio).lower()
            print(f"You said: {command}")
            return command
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            self.speak("Sorry, I couldn't connect to the speech recognition service.")
            return ""
            
    def process_command(self, command):
        for key, func in self.commands.items():
            if key in command:
                func()
                return True
        return False
        
    def show_camera_feed(self):
        if not self.show_camera:
            self.show_camera = True
            self.camera_thread = threading.Thread(target=self._camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            self.speak("Showing camera feed")
            
    def hide_camera_feed(self):
        if self.show_camera:
            self.show_camera = False
            if self.camera_thread:
                self.camera_thread.join()
            cv2.destroyAllWindows()
            self.speak("Hiding camera feed")
            
    def _camera_loop(self):
        while self.show_camera:
            ret, frame = self.camera.read()
            if not ret:
                break
                
            # Face detection and recognition
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"
                
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
                    
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                
            # Object detection
            results = self.model(frame)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    cls = box.cls[0]
                    if conf > 0.5:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        cv2.putText(frame, f"{self.model.names[int(cls)]} {conf:.2f}", 
                                  (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            
            cv2.imshow('Jarvis Vision', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    def identify_faces(self):
        if not self.show_camera:
            self.speak("Please show the camera feed first")
            return
            
        ret, frame = self.camera.read()
        if ret:
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            
            if not face_locations:
                self.speak("I don't see any faces")
                return
                
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"
                
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
                    
                self.speak(f"I see {name}")
                
    def detect_objects(self):
        if not self.show_camera:
            self.speak("Please show the camera feed first")
            return
            
        ret, frame = self.camera.read()
        if ret:
            results = self.model(frame)
            detected_objects = set()
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = box.conf[0]
                    cls = box.cls[0]
                    if conf > 0.5:
                        detected_objects.add(self.model.names[int(cls)])
                        
            if detected_objects:
                self.speak(f"I see {', '.join(detected_objects)}")
            else:
                self.speak("I don't see any objects")
                
    def exit_program(self):
        self.hide_camera_feed()
        self.camera.release()
        cv2.destroyAllWindows()
        self.speak("Goodbye!")
        exit()
        
    def run(self):
        self.speak("Hello! I'm Jarvis. How can I help you?")
        
        while True:
            command = self.listen()
            if command:
                if not self.process_command(command):
                    self.speak("I don't understand that command. Please try again.")
                    
if __name__ == "__main__":
    jarvis = Jarvis()
    jarvis.run() 