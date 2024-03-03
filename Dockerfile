FROM nvidia/cuda:12.3.1-devel-ubuntu22.04 as base

ARG USERNAME=python
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND=noninteractive
ENV JUPYTER_PLATFORM_DIRS=1

RUN apt-get update \
    && apt-get install -y \
    python3-pip \
    python3-dev \
    git-all \
    nano \
    && cd /usr/local/bin \
    && ln -s /usr/bin/python3 python \
    && pip3 --no-cache-dir install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

EXPOSE 8080

WORKDIR /src

RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    && mkdir /commandhistory \
    && touch /commandhistory/.bash_history \
    && chown -R $USERNAME /commandhistory \
    && echo "$SNIPPET" >> "/home/$USERNAME/.bashrc"

RUN pip install --upgrade pip

USER $USERNAME

CMD [ "python", "manage.py", "runserver" ]

FROM base as devl

WORKDIR /bird-species-image-classification

COPY ./src/requirements.txt /bird-species-image-classification
RUN python -m pip install -r requirements.txt

ENV PYTHONPATH /bird-species-image-classification

CMD [ "tail", "-f", "/dev/null" ]

FROM base as test

COPY ./src/requirements.txt /src
RUN python -m pip install -r requirements.txt
COPY ./src /src

ENV PYTHONPATH /src

FROM base as stage

COPY ./src/requirements.txt /src
RUN python -m pip install -r requirements.txt
COPY ./src /src

ENV PYTHONPATH /src

FROM base as prod

COPY ./src/requirements.txt /src
RUN python -m pip install -r requirements.txt
COPY ./src /src

ENV PYTHONPATH /src
