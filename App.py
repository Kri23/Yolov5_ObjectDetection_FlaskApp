from flask import Flask, request, jsonify, render_template
import requests
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import base64
import io

# UI Backend Service
ui_app = Flask(__name__)
CORS(ui_app)  # Enable Cross-Origin Resource Sharing

@ui_app.route('/')
def index():
    return render_template('index.html')

@ui_app.route('/upload', methods=['POST'])
def upload_image():
    try:
        # Retrieve the uploaded image
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image = request.files['image']
        if image.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Send the image to the AI backend service
        files = {'image': (image.filename, image.stream, image.mimetype)}
        ai_response = requests.post('http://localhost:5001/detect', files=files)

        if ai_response.status_code != 200:
            return jsonify({'error': 'AI backend error', 'details': ai_response.json()}), 500

        # Return the detections along with the base64 image
        return jsonify(ai_response.json()), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# AI Backend Service
ai_app = Flask(__name__)

# Load YOLO model (Using YOLOv5 PyTorch Hub implementation as an example)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

@ai_app.route('/detect', methods=['POST'])
def detect_objects():
    try:
        # Retrieve the uploaded image
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image = request.files['image']
        if image.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Read the image file
        image_data = Image.open(image.stream).convert('RGB')

        # Perform detection
        results = model(image_data)
        print(results)
        
        # Extracting predictions
        # The results object contains a 'xyxy' attribute (bounding boxes), 'conf' (confidence), and 'names' (class labels)
        predictions = results.xyxy[0]  # Get the first image's detections
        bboxes = predictions[:, :4]  # Extract bounding box coordinates (xmin, ymin, xmax, ymax)
        confidence = predictions[:, 4]  # Extract confidence scores
        class_ids = predictions[:, 5]  # Extract class IDs (index 5 corresponds to class ID)

        # Convert class IDs to class names
        labels = [model.names[int(class_id)] for class_id in class_ids]  # `model.names` maps class ID to class label

        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(image_data)
        for bbox, conf, label in zip(bboxes, confidence, labels):
            xmin, ymin, xmax, ymax = bbox
            draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=3)
            draw.text((xmin, ymin), f'{label} {conf:.2f}', fill='red')

        # Convert the image to base64
        buffered = io.BytesIO()
        image_data.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Format the results as JSON
        detection_results = [
            {
                'label': label,
                'confidence': conf.item(),  # Convert to Python float
                'bbox': {
                    'xmin': bbox[0].item(),  # Convert to Python float
                    'ymin': bbox[1].item(),
                    'xmax': bbox[2].item(),
                    'ymax': bbox[3].item()
                }
            }
            for bbox, conf, label in zip(bboxes, confidence, labels)
        ]

        # Return both the detection results and the base64-encoded image
        return jsonify({'detections': detection_results, 'image': img_str}), 200  

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def run_ui():
    ui_app.run(host='0.0.0.0', port=5000)

def run_ai():
    ai_app.run(host='0.0.0.0', port=5001)


if __name__ == '__main__':
    from multiprocessing import Process

    # Start the UI and AI backends as separate processes
    ui_process = Process(target=run_ui)
    ai_process = Process(target=run_ai)

    ui_process.start()
    ai_process.start()

    ui_process.join()
    ai_process.join()
