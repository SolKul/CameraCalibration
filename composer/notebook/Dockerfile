FROM camera_calibration:develop

USER jovyan
WORKDIR /home/jovyan
RUN mkdir /home/jovyan/works
RUN jupyter notebook --generate-config
RUN echo 'c.NotebookApp.ip = "0.0.0.0"\n\
c.NotebookApp.notebook_dir = "/home/jovyan/works"\n\
c.NotebookApp.password = "sha1:c0bd149d6e69:a7c58472d5a958f293865d4edf4ed564b8d4b2de"\n\
c.NotebookApp.open_browser = False'\
 >> /home/jovyan/.jupyter/jupyter_notebook_config.py

USER root
RUN echo '#!/bin/sh\n\
\n\
sudo -u jovyan jupyter notebook'\
 >> /home/jovyan/start-notebook.sh
RUN chmod 755 /home/jovyan/start-notebook.sh

CMD ["/home/jovyan/start-notebook.sh"]

# RUN echo 'c.NotebookApp.password = "sha1:c0bd149d6e69:a7c58472d5a958f293865d4edf4ed564b8d4b2de"' >> \
#  /home/jovyan/.jupyter/jupyter_notebook_config.py
 
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
