FROM 3.12.12-bookworm

WORKDIR /app

COPY . /app

RUN pip install -r requirements.text

CMD ["python3", "app.py"]