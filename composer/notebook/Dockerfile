FROM jupyter/datascience-notebook

USER root

RUN echo 'c.NotebookApp.password = "sha1:c0bd149d6e69:a7c58472d5a958f293865d4edf4ed564b8d4b2de"' >> \
 /home/jovyan/.jupyter/jupyter_notebook_config.py
 
RUN mkdir /home/jovyan/notebooks
RUN chown jovyan:users /home/jovyan/notebooks

RUN apt-get update

USER jovyan

#jovyanでpip、condaできるように、/opt/conda/binをPATHに追加
RUN echo 'export PATH=$PATH:/opt/conda/bin' >> ~/.bashrc

RUN pip install jupyter-contrib-nbextensions
RUN jupyter contrib nbextension install --user 

RUN pip install jupytext

RUN echo 'c.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager"' >> \
 /home/jovyan/.jupyter/jupyter_notebook_config.py
RUN echo 'c.ContentsManager.default_jupytext_formats = "ipynb,py"' >> \
 /home/jovyan/.jupyter/jupyter_notebook_config.py
 

WORKDIR /home/jovyan

USER root
