FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

COPY pyproject.toml pyproject.toml

RUN pip install poetry

RUN poetry config virtualenvs.create false

RUN poetry install --no-dev 

COPY ./src /app

CMD ["/start-reload.sh"]
