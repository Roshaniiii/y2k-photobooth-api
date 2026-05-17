from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
import numpy as np

from filters.vhs import apply_vhs
from filters.glitch import apply_glitch
from filters.y2k import apply_y2k
from filters.crt import apply_crt
from filters.grain import apply_grain
from filters.chroma import apply_chroma

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FilterRequest(BaseModel):
    image: str
    filter: str

def decode_image(b64_string: str):
    image_data = base64.b64decode(b64_string)
    image_array = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)

def encode_image(img) -> str:
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/apply-filter")
async def apply_filter(request: FilterRequest):
    img = decode_image(request.image)

    if request.filter == "vhs":
        img = apply_vhs(img)
    elif request.filter == "glitch":
        img = apply_glitch(img)
    elif request.filter == "y2k":
        img = apply_y2k(img)
    elif request.filter == "crt":
        img = apply_crt(img)
    elif request.filter == "grain":
        img = apply_grain(img)
    elif request.filter == "chroma":
        img = apply_chroma(img)

    return {"image": encode_image(img)}

@app.get("/")
async def root():
    return {"status": "Y2K Photobooth API is running!"}