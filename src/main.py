from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import io
import cv2
import numpy as np
from PIL import Image
import base64
from trash_can_model import TrashCanModel

app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the model
model = TrashCanModel()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse("index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Make prediction
    detections, image_with_boxes = model.predict(image)
    
    # Encode the image with bounding boxes to base64
    _, buffer = cv2.imencode('.jpg', image_with_boxes)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return JSONResponse({
        "predictions": detections,
        "image": image_base64
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)