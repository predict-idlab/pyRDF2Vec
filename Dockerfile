FROM jupyter/datascience-notebook
LABEL maintainer="IDLab"
LABEL description="Build Image for Local Development with Jupyter."

ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_NO_CACHE_DIR=off \
    POETRY_NO_INTERACTION=1 \
    POETRY_VERSION=1.1.8 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1

RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /home/jovyan/work

# Cache the requirements and reinstall them only when a change occurs in the
# pyproject.toml files
COPY poetry.lock pyproject.toml /home/jovyan/work/

RUN poetry config virtualenvs.create false \
    && poetry install --

COPY . /code
