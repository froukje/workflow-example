FROM python:3.10-slim

WORKDIR WORK

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY apply-model.py .
CMD ["python3", "apply-model.py"]
