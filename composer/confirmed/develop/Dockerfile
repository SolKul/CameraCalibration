#開発環境に必要なもの
FROM camera_calibration:base

USER root

RUN apt update
RUN apt install sudo=1.8.27-1

RUN useradd -D -s /bin/bash
RUN useradd -m jovyan
WORKDIR /home/jovyan

RUN usermod -G sudo jovyan
RUN echo 'jovyan ALL=NOPASSWD: ALL\n\
Defaults        secure_path="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/conda/bin:/opt/conda/condabin"'\
>> /etc/sudoers

RUN chown jovyan:jovyan -R /opt/conda

USER root

USER jovyan
RUN conda install -y jupyter=1.0.0=py37_7