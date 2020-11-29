FROM python:3.6

WORKDIR /

COPY requirements.txt .

COPY ./ocr_manga .

RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "main.py" ]

EXPOSE 8080