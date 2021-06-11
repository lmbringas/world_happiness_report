import os
import time

from fastapi import FastAPI, File, UploadFile

app = FastAPI()


@app.get("/")
async def hello_word():
    return {"status": "okey"}


@app.post("/upload_report/")
async def image(dataset: UploadFile = File(...)):
    print(dataset)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = f"{dataset.filename}"
    f = open(f"{filename}", "wb")
    content = await dataset.read()
    f.write(content)
    return {"okey": "polilla"}
