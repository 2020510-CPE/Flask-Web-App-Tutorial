from flask import Blueprint, render_template, request, flash, jsonify
from flask_login import login_required, current_user
from .models import Note
from . import db
import json
from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
from time import sleep



views = Blueprint('views', __name__)
# Load YOLO model
model = YOLO('best4_13_24.pt')
print("MODEL WAS LOADED!")

offset = 10  # Allowed error between pixels

# Open camera
cap = cv2.VideoCapture(1)
line_position = 400  # Position of the counting line
def compute_center(coordinates):
    x_center = (coordinates[0] + coordinates[2]) / 2
    y_center = (coordinates[1] + coordinates[3]) / 2
    return x_center, y_center

def generate_frames():
    fps = 60  
    count = 0 
    counted_ids = set()  

    while True:
        success, frame = cap.read()
        time = float(1 / fps)
        sleep(time)
        if not success:
            break
        else:
            results = model.track(frame, persist=True, conf=0.10)
            result = results[0]  
            if len(result.boxes) > 0: 
                for box in result.boxes:

                    cords = box.xyxy[0].tolist()
                    cords = [round(x) for x in cords]
                    class_id = result.names[box.cls[0].item()]
                    #print("ETOOO")
                    #if box.id is not None:
                        #print(box.id.item())   
                    conf = round(box.conf[0].item(), 2)

                    # Obtain object ID 
                    if box.id is not None:
                        object_id = box.id.item() 


                    # Calculate center of bounding box
                    x_center, y_center = compute_center(cords) 

                    if object_id not in counted_ids:
                        # Check if the object crossed the line
                        if y_center < (line_position + offset) and y_center > (line_position - offset):
                            count += 1
                            counted_ids.add(object_id)  

                    # Visualization 
                    cv2.rectangle(frame, (int(cords[0]), int(cords[1])), (int(cords[2]), int(cords[3])), (255,0,0), 2)
                    
                    cv2.putText(frame, "Count: " + str(count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  

            # Draw counting line 
            cv2.line(frame, (25, line_position), (1200, line_position), (255, 127, 0), 3) 

            frame = results[0].plot() 
            #print(results[0].plot()) 
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                
@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    if request.method == 'POST': 
        note = request.form.get('note')#Gets the note from the HTML 

        if len(note) < 1:
            flash('Note is too short!', category='error') 
        else:
            new_note = Note(data=note, user_id=current_user.id)  #providing the schema for the note 
            db.session.add(new_note) #adding the note to the database 
            db.session.commit()
            flash('Note added!', category='success')

    return render_template("home.html", user=current_user)


@views.route('/delete-note', methods=['POST'])
def delete_note():  
    note = json.loads(request.data) # this function expects a JSON from the INDEX.js file 
    noteId = note['noteId']
    note = Note.query.get(noteId)
    if note:
        if note.user_id == current_user.id:
            db.session.delete(note)
            db.session.commit()

    return jsonify({})

@views.route('/')
def index():
    return render_template('index.php')

@views.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
