FROM continuumio/miniconda3:4.7.10

#開発環境と本番で同じところ
#必要なモジュールなど
USER root

RUN apt update
RUN apt install -y libgl1-mesa-dev=18.3.6-2 

RUN conda install -y -c conda-forge opencv=4.1.1=py37hd64ca61_0

USER root