from flask import Flask, render_template, request, jsonify, send_from_directory
from flask import redirect, url_for
import os
from pathlib import Path
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = './static/uploads'
DETECT_FOLDER = 'C:/projects/sesac/runs/detect/predict2'  # 수정된 폴더 경로

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECT_FOLDER'] = DETECT_FOLDER

DETECT_RESULT = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'images' in request.files:
            images = request.files.getlist('images')
            for i, image in enumerate(images):
                image_filename = f"image_{i + 1}.jpg"
                image.save(os.path.join(app.config['UPLOAD_FOLDER'], image_filename))
            result = '이미지 전송 및 저장 성공'
        else:
            result = '이미지를 선택하세요.'
    except Exception as e:
        result = f'오류 발생: {str(e)}'
    return jsonify({'result': result})

@app.route('/display')
def display():
    image_data = []
    upload_folder_path = Path(app.config['UPLOAD_FOLDER'])
    detect_folder_path = Path(app.config['DETECT_FOLDER'])

    for image_path in upload_folder_path.glob('*.jpg'):
        image_filename = image_path.name
        description_filename = f"{image_filename.split('.')[0]}.txt"
        try:
            with open(detect_folder_path / description_filename, 'r') as file:
                descriptions = file.read().splitlines()
        except FileNotFoundError:
            descriptions = []

        image_data.append({'filename': image_filename, 'descriptions': descriptions})

    return render_template('display.html', image_data=image_data)

@app.route('/yolo')
def yolo():
    try:
        model = YOLO('yolov8n.pt')
        results = model.predict(UPLOAD_FOLDER, task='detect', save=True)
        DETECT_RESULT.append(results[0].tojson())
        return redirect(url_for('display'))
    except Exception as e:
        return render_template('error.html', error=str(e))

# 새로운 라우트 추가: 이미지 및 설명 출력
@app.route('/object_detection/<filename>')
def object_detection(filename):
    return send_from_directory(app.config['DETECT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
