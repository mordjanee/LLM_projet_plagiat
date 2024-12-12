import json
import logging
import os
from asyncio import Task
from typing import List, Dict

from sanic import Sanic, response, HTTPResponse
from sanic.exceptions import SanicException
from sanic.request import Request

from plagiat import Plagiat



app = Sanic("Extract_real_pdf")
app.config.RESPONSE_TIMEOUT = 6000
app.config.KEEP_ALIVE_TIMEOUT = 6000
app.config.REQUEST_MAX_SIZE = 300_000_000




@app.post("/detect-plagiat")
def upload_pdf(request: Request) -> HTTPResponse:
    if not request.files:
        raise SanicException("error : No 'files' in your form data body", 500)

    file: Dict[str, str] = request.files.get('file')
    if not file:
        raise SanicException("can not find a pdf file", 500)
    if file.type != "application/pdf":
        if file.type not in ["application/pdf", "text/plain"]:
            raise SanicException("File is not a PDF or TXT", 415)

    try:
        detect_plagiat = Plagiat()
        if file.type == "application/pdf":
            filetype = 'pdf'
        elif file.type == "text/plain":
            filetype = 'txt'
        result = detect_plagiat.detect_plagiat(file, filetype)
    except Exception as e:
        raise SanicException(f"Error processing extract pdf: {e}", 500)
    return response.json(result)




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, single_process=True)
    logging.info("SERVER IS RUNNING")