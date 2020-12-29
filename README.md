# CameraCalibration
基本部分のimageをbaseで構築  
必要なモジュールなど  

開発と運用は分けたほうがいいと結論
勉強しながらコード書いていく環境と
実際に運用する場は別

## 起動方法

- dockerを起動
- Kitematicを起動
- 当該コンテナを起動
- localhost:8889にアクセス
- Kitematicで共有化しているフォルダを調べ、VSCodeで開く
- Gitの更新を管理する


正直めんどくさすぎるからdevcontainerに移行したほうがいいと思う。

## 環境構築
`/composer/confirmed`で`docker-compose build`で`camera_calibration:base`というimageを作成  
  
つぎに`/composer`で`docker-compose up -d`で
開発環境をたちあげ

そして`docker attach camera_calibration`で開発環境に入る

開発環境の際にはmy_env.txtでハッシュ化したパスワードを環境変数とし、
`start-notebook.sh --NotebookApp.password=$NotebookApp_password`としてパスワードを指定

## モジュールのインストール
まず`conda search`でバージョンを調べる。
```bash
conda search -c conda-forge --f jupytext
```
次にバージョンを指定してインストール
```bash
sudo -u jovyan conda install -c conda-forge jupytext=1.2.4=0

echo $'c.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager"\n
c.ContentsManager.default_jupytext_formats = "ipynb,py"'\
>> /home/jovyan/.jupyter/jupyter_notebook_config.py
```

うまく動いたらdokcerfileに反映
```Dockerfile
RUN conda install -y -c conda-forge jupytext=1.2.4=0
RUN echo 'c.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager"\n\
c.ContentsManager.default_jupytext_formats = "ipynb,py"'\
>> /home/jovyan/.jupyter/jupyter_notebook_config.py
```

### jupyter_contrib_nbextensions
jupyter_contrib_nbextensionsはjovyan権限でインストールしないとだめらしい。  
jovyan権限でインストールするため`sudo -u jovyan`としている。
```
sudo -u jovyan conda install -c conda-forge jupyter_contrib_nbextensions=0.5.1=py37_0  
sudo -u jovyan jupyter contrib nbextension install --user 
```

```Dockerfile
USER jovyan
RUN conda install -y -c conda-forge jupyter_contrib_nbextensions=0.5.1=py37_0
RUN jupyter contrib nbextension install --user
```


### opencv

```bash
conda search -c conda-forge opencv
sudo -u jovyan conda install -c conda-forge opencv=4.1.1=py37hd64ca61_0
```

```Dockerfile
RUN conda install -y -c conda-forge opencv=4.1.1=py37hd64ca61_0
```