# CameraCalibration
基本部分のimageをconfirmedで構築  
必要なモジュールなど  
```
docker build -t camera_calibration:base confirmed/base  
docker build -t camera_calibration:develop confirmed/develop   
```

docker run --rm --name anaconda_test -it confirmed:1 /bin/bash  

sudo -u jovyan conda install -c conda-forge jupyter_contrib_nbextensions=0.5.1=py37_0  
付随する開発環境はあとから入れるといいかも  
docker-compose up  

そうすればjupyterで開発終わった後、  
基本のimageでアプリケーションとしてできるかも。  

開発と運用は分けたほうがいいと結論
勉強しながらコード書いていく環境と
実際に運用する場は別