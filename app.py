import argparse
import os
import warnings
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import pickle
import PIL.Image
import torch
from PIL import Image
from tqdm import tqdm
from fastapi import FastAPI, HTTPException, Body, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
import base64
from io import BytesIO
from typing_extensions import Annotated

from fastapi import Depends, FastAPI
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from mangum import Mangum
import boto3
import json
from datetime import datetime
import random
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import base64
import cv2
import numpy as np
import torch
import uvicorn 
from lib.model_zoo.migan_inference import Generator as MIGAN
from lib.model_zoo.comodgan import (
    Generator as CoModGANGenerator,
    Mapping as CoModGANMapping,
    Encoder as CoModGANEncoder,
    Synthesis as CoModGANSynthesis
)

warnings.filterwarnings("ignore")
def save_base64_to_s3(base64_data, bucket_name, file_name):
    s3 = boto3.client(
        's3',
        aws_access_key_id="AKIASPWDOKNSB44BNA7P",
        aws_secret_access_key="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" # Add full S3 access to these keys
    )
    decoded_data = base64.b64decode(base64_data)

    try:
        s3.put_object(Body=decoded_data, Bucket=bucket_name, Key=file_name, ACL='public-read')
        
        #url = s3.generate_presigned_url(
        #ClientMethod='get_object', 
        #Params={'Bucket': bucket_name, 'Key': file_name},
        #ExpiresIn=21600)
        
        object_url = f"https://{bucket_name}.s3.amazonaws.com/{file_name}"
        return object_url
    except Exception as e:
        return f'Error uploading to S3: {str(e)}'


def s3_uploader(base64_data):
    body = base64_data
    # Generate a random timestamp within a reasonable range
    random_timestamp = datetime.fromtimestamp(random.randint(0, 2**31-1))
    
    # Format the timestamp as a string
    formatted_timestamp = random_timestamp.strftime("%Y%m%d%H%M%S")
    
    # Generate a random number to add uniqueness
    random_number = random.randint(1000, 9999)
    bucket_name = "ai-processed-images"  # Replace with your specific S3 bucket name
    file_name = f"uploaded-image-{formatted_timestamp}-{random_number}.png"  # Replace with your desired file name

    object_url = save_base64_to_s3(body, bucket_name, file_name)
    return object_url

def read_mask(mask_path, invert=False):
    mask = Image.open(mask_path)
    mask = resize(mask, max_size=512, interpolation=Image.NEAREST)
    mask = np.array(mask)
    if len(mask.shape) == 3:
        if mask.shape[2] == 4:
            _r, _g, _b, _a = np.rollaxis(mask, axis=-1)
            mask = np.dstack([_a, _a, _a])
        elif mask.shape[2] == 2:
            _l, _a = np.rollaxis(mask, axis=-1)
            mask = np.dstack([_a, _a, _a])
        elif mask.shape[2] == 3:
            _r, _g, _b = np.rollaxis(mask, axis=-1)
            mask = np.dstack([_r, _r, _r])
    else:
        mask = np.dstack([mask, mask, mask])
    if invert:
        mask = 255 - mask
    mask[mask < 255] = 0
    return Image.fromarray(mask).convert("L")


def resize(image, max_size, interpolation=Image.BICUBIC):
    w, h = image.size
    if w > max_size or h > max_size:
        resize_ratio = max_size / w if w > h else max_size / h
        image = image.resize((int(w * resize_ratio), int(h * resize_ratio)), interpolation)
    return image


def preprocess(img: Image, mask: Image, resolution: int) -> torch.Tensor:
    img = img.resize((resolution, resolution), Image.BICUBIC)
    mask = mask.resize((resolution, resolution), Image.NEAREST)
    img = np.array(img)
    mask = np.array(mask)[:, :, np.newaxis] // 255
    img = torch.Tensor(img).float() * 2 / 255 - 1
    mask = torch.Tensor(mask).float()
    img = img.permute(2, 0, 1).unsqueeze(0)
    mask = mask.permute(2, 0, 1).unsqueeze(0)
    x = torch.cat([mask - 0.5, img * mask], dim=1)
    return x


resolution = 512
model = MIGAN(resolution=512)
model.load_state_dict(torch.load("migan_512_places2.pt", map_location=torch.device('cpu')))
model.eval()


app = FastAPI()
handler = Mangum(app)



from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:5175",
      "ANY_OR_ALL_front_end_url"  # Update this to your frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


api_key_1 = "a1b2c3d4e5"



class ImageInput(BaseModel):
    image_base64: str
    mask_base64: str 
@app.post("/process_images")
async def process_images(image: ImageInput, key: str = Body(embed=True)):
    if key != api_key_1:
        raise HTTPException(status_code=401, detail="Invalid API key")
    try:


        img_path = BytesIO(base64.b64decode(image.image_base64))
        mask_path = BytesIO(base64.b64decode(image.mask_base64))

        img = Image.open(img_path).convert("RGB")
        resolution = 512
        img_resized = resize(img, max_size=resolution)
        mask = read_mask(mask_path, invert=True)
        mask_resized = resize(mask, max_size=resolution, interpolation=Image.NEAREST)

        x = preprocess(img_resized, mask_resized, resolution)
        #if cuda:
        #    x = x.to("cuda")
        with torch.no_grad():
            result_image = model(x)[0]
        result_image = (result_image * 0.5 + 0.5).clamp(0, 1) * 255
        result_image = result_image.to(torch.uint8).permute(1, 2, 0).detach().to("cpu").numpy()

        result_image = cv2.resize(result_image, dsize=img_resized.size, interpolation=cv2.INTER_CUBIC)
        mask_resized = np.array(mask_resized)[:, :, np.newaxis] // 255
        composed_img = img_resized * mask_resized + result_image * (1 - mask_resized)
        composed_img = Image.fromarray(composed_img)
        #composed_img.save("7.PNG")



        im_file = BytesIO()



        composed_img.save(im_file, format="PNG")

        im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
        im_b64 = base64.b64encode(im_bytes)
        public_url = s3_uploader(im_b64)
        #return public_url
        #print(public_url)
        #composed_img.save("91.png")

        #im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
        #im_b64 = base64.b64encode(im_bytes)
        #print(im_b64)
        return JSONResponse(content={"message": "Image processed successfully", "result_image": public_url})
    except Exception as e:
        return JSONResponse(content={"error": f"Error processing image: {str(e)}"}, status_code=500)
if __name__ == "__main__":
#   uvicorn.run(app)
   uvicorn.run(app, host="127.0.0.1", port=8080)
        
        

