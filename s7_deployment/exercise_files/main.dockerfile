FROM python:3.9
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./app /code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

# docker build -t my_fastapi_app .

# docker run --name mycontainer -p 80:80 myimage

# http://localhost/items/5?q=somequery