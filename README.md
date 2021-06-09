# Waymo Open Dataset 開發紀錄
contributed by < `gyes00205` >
###### tags: `waymo`

## Download Dataset
[Waymo Open Dataset](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0)
domain_adaptation 的資料沒有 label ，所以請下載 training/ 的資料去訓練

## tfrecord 內容
* 五種 camera 所拍攝的照片以及 Lidar 資訊
* num_classes: 0: Unknown, 1: Vehicle, 2: Pedestrian, 3: Sign, 4: Cyclist 
    這次 project 不需要 sign 和 Unknown 這兩個 classes ，因此 label_map.pbtxt 修改如下:
```pbtxt 
item {
    id: 1
    name: 'vehicle'
}

item {
    id: 2
    name: 'pedestrian'
}

item {
    id: 4
    name: 'cyclist'
}
```
* camera 種類: FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT
    <img src="https://i.imgur.com/Q68Lepf.jpg">
* bbox (x, y, w, h) 座標: , xy 代表 bbox 中心座標 , wh 代表寬和高

## 環境配置
### 安裝 Way open dataset
```shell 
pip3 install waymo-open-dataset-tf-2-1-0==1.2.0
```
### 安裝 COCO API
```shell 
pip install cython
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```
### 安裝 Tensorflow 2 Object Detection API
參考 [TensorFlow 2 Object Detection API tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html) 安裝套件
* git clone Tensorflow 2 Object Detection API
```shell 
git clone https://github.com/tensorflow/models.git
```
* 到 models/research/ 執行
```shell 
protoc object_detection/protos/*.proto --python_out=.
```
* 將 API 加到環境變數
```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
* 複製 setup.py 到 models/research/
```shell 
cp object_detection/packages/tf2/setup.py ./
```
* 安裝 setup.py
```shell 
python -m pip install .
```
* 測試是否安裝成功
```
python object_detection/builders/model_builder_tf2_test.py
```