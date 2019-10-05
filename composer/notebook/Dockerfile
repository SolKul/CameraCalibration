FROM camera_calibration:develop

USER root
# RUN apt install -y libgl1-mesa-dev/=18.3.6-2 
# RUN apt install -y vim=2:8.1.0875-5

# USER jovyan
# RUN conda install -y -c conda-forge jupyter_contrib_nbextensions=0.5.1=py37_0
# RUN jupyter contrib nbextension install --user

#sudo -u jovyan conda install -c conda-forge jupyter_contrib_nbextensions=0.5.1=py37_0  
#sudo -u jovyan jupyter contrib nbextension install --user 

# RUN conda install -y -c conda-forge jupytext=1.2.4=0
# RUN echo 'c.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager"\n\
# c.ContentsManager.default_jupytext_formats = "ipynb,py"'\
# >> /home/jovyan/.jupyter/jupyter_notebook_config.py

#conda search -c conda-forge --f jupytext
#sudo -u jovyan conda install -c conda-forge jupytext=1.2.4=0
# echo $'c.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager"\n
# c.ContentsManager.default_jupytext_formats = "ipynb,py"'\
# >> /home/jovyan/.jupyter/jupyter_notebook_config.py

# RUN conda install -y -c conda-forge opencv=4.1.1=py37hd64ca61_0

# conda search -c conda-forge opencv
#sudo -u jovyan conda install -c conda-forge opencv=4.1.1=py37hd64ca61_0

# CMD ["/home/jovyan/start-notebook.sh"]



CMD ["/bin/bash"]
USER root
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
