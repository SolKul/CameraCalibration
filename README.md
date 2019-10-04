# CameraCalibration
基本部分のimageをconfirmedで構築
必要なモジュールなど
docker build -t confirmed:base confirmed/base
docker build -t confirmed:update confirmed/update

docker run --rm --name anaconda_test -it confirmed:1 /bin/bash

付随する開発環境はあとから入れるといいかも
docker-compose up

そうすればjupyterで開発終わった後、
基本のimageでアプリケーションとしてできるかも。
