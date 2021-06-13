import threading

from fastapi import FastAPI, File, UploadFile

from processing.run_pipeline import run_pipeline

app = FastAPI()


@app.get("/")
async def hello_word():
    return {"status": "okey"}


@app.post("/upload_report/")
async def image(dataset: UploadFile = File(...)):
    pipeline = threading.Thread(target=run_pipeline, args=(dataset,), daemon=True)
    pipeline.start()
    pipeline.join()
    return {"okey": "polilla"}
