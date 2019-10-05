FROM camera_calibration:base

USER root
RUN apt install -y vim

USER jovyan
RUN conda install -y matplotlib=3.1.1=py37h5429711_0 scipy=1.3.1=py37he2b7bc3_0

CMD ["/bin/bash"]
ENTRYPOINT [ "" ]
USER root
 
# RUN mkdir /home/jovyan/notebooks
# RUN chown jovyan:users /home/jovyan/notebooks

# RUN apt-get update

# USER jovyan

# #jovyanでpip、condaできるように、/opt/conda/binをPATHに追加
# RUN echo 'export PATH=$PATH:/opt/conda/bin' >> ~/.bashrc

# RUN pip install jupyter-contrib-nbextensions
# RUN jupyter contrib nbextension install --user 

# RUN pip install jupytext

# RUN echo 'c.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager"' >> \
#  /home/jovyan/.jupyter/jupyter_notebook_config.py
# RUN echo 'c.ContentsManager.default_jupytext_formats = "ipynb,py"' >> \
#  /home/jovyan/.jupyter/jupyter_notebook_config.py
 

# WORKDIR /home/jovyan

# USER root
