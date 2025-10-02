from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
import os

app = FastAPI()


@app.post("/analyze")
async def analyze(
    text: str = Form(None),
    file: UploadFile = File(None)
):
    """
    Accept EITHER text OR file (but not both).
    """

    if text and file:
        return {"error": "Please provide only text OR file, not both."}

    if text:
        # TODO: plug into your compliance grader
        return {
            "input_type": "text",
            "length": len(text),
            "preview": text[:100],
            "status": "processed"
        }

    if file:
        save_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        with open(save_path, "wb") as buffer:
            buffer.write(await file.read())

        return {
            "input_type": "file",
            "filename": file.filename,
            "path": save_path,
            "status": "uploaded"
        }

    return {"error": "No input provided. Please send text or a file."}
