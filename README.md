# 🐦 FastAPI Microservice for Bird Species Classification

This project provides an example of a [FastAPI](https://fastapi.tiangolo.com/) microservice which hosts a bird species classification model from Kaggle's [Birds 525 Species - Image CLassification](https://www.kaggle.com/datasets/gpiosenka/100-bird-species/data) dataset. This project provides a [VS Code development container configuration](https://code.visualstudio.com/docs/devcontainers/containers) which is the recommended method for developing this project.

This project is almost completely typed and leverages Pydantic for type validation.

This project leverages the following repositories and works:

- [fastapi-realworld-example-app](https://github.com/nsidnev/fastapi-realworld-example-app/tree/master)
- []

## Prerequisites

This project assumes common knowledge and understanding of VS Code and the [Remote Development Extension Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) for VS Code.

This project is also configured to utilize NVIDIA's `nvidia/cuda` Docker image, which will enable use of NVIDIA GPUs for this project. This project does not require an NVIDIA GPU, but it is configured to take advantage of an NVIDIA GPU if available.

## Quick Start

The following instructions provide a quick start for developing `fastapi-microservice-bird-species-classification` on _Linux_ or _WSL_.

### 1. Clone repository

```bash
git clone https://github.com/joshjhans/fastapi-microservice-bird-species-classification.git
```

### 2. Attach VS Code to cloned `fastapi-microservice-bird-species-classification` directory

### 3. Open `fastapi-microservice-bird-species-classification` directory in a container

![VS Code Open Folder in Container](src/docs/static/open-folder-in-container.png)

### 3. Open pre-configured terminal

Press <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>B</kbd> to open a pre-configured terminal in the `/bird-species-image-class-fication/src` directory.

### 3. Download 'Birds 525 Species - Image Classification' dataset from [Huggingface](https://www.kaggle.com/datasets/gpiosenka/100-bird-species/data)

```bash
python manage.py download-dataset
```

### 4. Train the model

```bash
python manage.py train-model
```

### 4. Test the model

```bash
python manage.py test-model
```

### 4. Test API

```bash
python -m pytest .
```

### 5. Run the FastAPI service with `uvicorn`

```bash
python manage.py runserver
```

### 6. Navigate to API documentation

Navigate to [http://localhost:8080/api/docs](http://localhost:8080/api/docs)

### 7. Upload an image of one of the 525 bird species
