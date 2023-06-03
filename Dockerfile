FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

RUN apt-get update \
    && apt-get install -y \
        python3-pip \
        python-is-python3 \
        flatbuffers-compiler \
        sudo \
    && pip install -U pip \
    && pip install -U tfliteiorewriter \
    && pip install -U tfliteiorewriter \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

ENV USERNAME=user
RUN echo "root:root" | chpasswd \
    && useradd \
        --create-home \
        --home-dir /home/${USERNAME} \
        --shell /bin/bash \
        --user-group \
        --groups adm,sudo \
        ${USERNAME} \
    && echo "${USERNAME}:${USERNAME}" | chpasswd \
    && cat /dev/null > /etc/sudoers.d/${USERNAME} \
    && echo "%${USERNAME}    ALL=(ALL)   NOPASSWD:    ALL" >> \
        /etc/sudoers.d/${USERNAME}
USER ${USERNAME}
WORKDIR /home/${USERNAME}