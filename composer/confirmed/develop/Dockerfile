#開発環境に必要なもの
FROM camera_calibration:base

USER root

RUN apt install -y sudo=1.8.27-1 vim=2:8.1.0875-5

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

RUN conda install -y -c conda-forge jupyter_contrib_nbextensions=0.5.1=py37_0 jupytext=1.2.4=0
RUN jupyter contrib nbextension install --user

WORKDIR /home/jovyan
RUN mkdir /home/jovyan/works
RUN jupyter notebook --generate-config
RUN echo 'c.NotebookApp.ip = "0.0.0.0"\n\
c.NotebookApp.notebook_dir = "/home/jovyan/works"\n\
c.NotebookApp.password = "sha1:c0bd149d6e69:a7c58472d5a958f293865d4edf4ed564b8d4b2de"\n\
c.NotebookApp.open_browser = False\n\
c.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager"\n\
c.ContentsManager.default_jupytext_formats = "ipynb,py"'\
 >> /home/jovyan/.jupyter/jupyter_notebook_config.py

USER root
RUN echo '#!/bin/sh\n\
\n\
sudo -u jovyan jupyter notebook'\
 >> /home/jovyan/start-notebook.sh
RUN chmod 755 /home/jovyan/start-notebook.sh

CMD ["/home/jovyan/start-notebook.sh"]