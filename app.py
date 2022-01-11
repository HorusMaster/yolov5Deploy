import io
import torch
import math
import numpy as np

from PIL import Image
from flask import Flask, jsonify, request

app = Flask(__name__)
# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/best.pt')  # custom model
model.conf = 0.75  # confidence threshold (0-1)
model.iou = 0.45  # NMS IoU threshold (0-1)
#model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs


def get_prediction(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    results = model(image, size=608)  # includes NMS

    results.pandas().xyxy[0]  # img1 predictions (pandas)

    temp_df=results.pandas().xyxy[0] 
    distance = []
    max_iter = temp_df.shape[0]

    index_liters = temp_df[temp_df['class']==11]. index
    index_counter = temp_df[temp_df['class']==10]. index

    if len(index_liters) == 1 :
        for i in range(max_iter):
            dist_x1y1 = math.sqrt(math.pow((temp_df.iloc[i]['xmin'] - temp_df.iloc[index_liters]['xmin']), 2) + \
                                    math.pow((temp_df.iloc[i]['ymin'] - temp_df.iloc[index_liters]['ymin']), 2))

            distance.append(dist_x1y1)

        indexes_ordered = np.argsort(distance)[::-1]

    elif len(index_counter) == 1:
        for i in range(max_iter):
            dist_x1y1 = math.sqrt(math.pow((temp_df.iloc[i]['xmin'] - temp_df.iloc[index_counter]['xmin']), 2) + \
                                    math.pow((temp_df.iloc[i]['ymin'] - temp_df.iloc[index_counter]['ymin']), 2))

            distance.append(dist_x1y1)

        indexes_ordered = np.argsort(distance)
    else:
        indexes_ordered = temp_df.index

    class_ordered = [temp_df.iloc[index]['class'] for index in indexes_ordered]
    class_ordered = [str(digit) for digit in class_ordered if digit not in [10, 11]]
    return(int(''.join(class_ordered)))
     

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'lectura': class_name})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)