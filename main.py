import math
import cv2
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import numpy as np

app = FastAPI()

# Load YOLO model
model = YOLO("yolov8n.pt")

# Serve static files (for images)
app.mount("/static", StaticFiles(directory="static"), name="static")

className = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
             "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", 
             "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
             "suitcase", "frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove",
             "skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon",
             "bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
             "donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet",
             "tvmonitor","laptop","mouse","remote","keyboard","cell phone",
             "microwave","oven","toaster","sink","refrigerator",
             "book","clock","vase","scissors",
             "teddy bear","hair drier","toothbrush"]

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Image Upload</title>
        </head>
        <body>
            <h1>Upload an Image for Object Detection</h1>
            <form action="/upload/" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required>
                <input type="submit" value="Upload">
            </form>
        </body>
    </html>
    """

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Read the uploaded image file
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Process the image with YOLO model
    results = model(img, stream=True)
    
    # Draw bounding boxes on the image
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            label = f'{className[cls]} {conf}'
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, label, (x1, max(35, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Save the output image
    output_path = f'static/output.jpg'
    cv2.imwrite(output_path, img)

    return FileResponse(output_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
