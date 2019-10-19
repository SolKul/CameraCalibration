FROM jupyter/minimal-notebook:1386e2046833

#開発環境と本番で同じところ
# 必要なモジュールなど
USER root

RUN apt update
RUN apt install -y libgl1-mesa-dev=19.0.8-0ubuntu0~18.04.2


USER jovyan
RUN conda install -y -c conda-forge opencv=4.1.1=py37hd64ca61_0


#開発環境でのみ必要なもの
RUN conda install -y -c conda-forge jupyter_contrib_nbextensions=0.5.1=py37_0 jupytext=1.2.4=0
RUN jupyter contrib nbextension install --user
RUN jupyter nbextension enable highlight_selected_word/main
RUN echo 'c.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager"\n\
c.ContentsManager.default_jupytext_formats = "ipynb,py"'\
>> /home/jovyan/.jupyter/jupyter_notebook_config.py

USER root
# CMD [ "" ]
# ENTRYPOINT [ "" ]