from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
from fastapi import FastAPI
from fastapi import UploadFile, File
from fastapi.responses import FileResponse
from http import HTTPStatus
from typing import List
"""
curl -X 'POST' \
  'http://localhost:8000/input/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'files=@my_cat.jpg;type=image/jpeg' \
  -F 'files=@my_cat2.jpg;type=image/jpeg'
"""
app = FastAPI()

@app.get("/")
def root():
    """ Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

gen_kwargs = {"max_length": 16, "num_beams": 8, "num_return_sequences": 1}
@app.post("/input/")
async def predict_step(files: List[UploadFile]):
    image_paths = []
    for i, file in enumerate(files):
        with open(f'image{i}.jpg', 'wb') as image:
            content = await file.read()
            image.write(content)
            image.close()
            image_paths.append(f'image{i}.jpg')
            FileResponse('image.jpg')

    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    # return preds
    response = {
        "input": files,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "preds": preds
    }
    with open('output.txt', 'w') as f:
        f.write(str(preds))
    return response