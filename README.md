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
### 建立資料夾結構
```
Waymo
├───models #Tensorflow Object Detection API
├───training_configs #訓練用的 config
├───pre-trained-models #預訓練模型
├───exported-models #輸出模型
└───data #訓練資料
    └───segment-???.tfrecord
```
## 轉換 tfrecord 格式
因為 waymo 的 tfrecord 除了有 Lidar 的資訊之外，他的 bbox 格式如下:

(x0, y0): 為中心點座標。 (w, h): 為長寬。

![](https://i.imgur.com/WSDKAQZ.png)

而我們的目標是過濾掉 Lidar 並將 bbox 轉為以下格式:

(x1, y1): 為左上角座標。 (x2, y2): 為右下角座標。

![](https://i.imgur.com/HyR6xS0.png)

轉換 tfrecord 的程式碼參考 [LevinJ/tf_obj_detection_api](https://github.com/LevinJ/tf_obj_detection_api)，並且做一些小修改。

**create_record.py:**

filepath: tfrecord 的路徑

data_dir: 轉換過後的 tfrecord 會儲存在 data_dir/processed 目錄下

執行方式如下:
    
```shell 
python create_record.py \
--filepath=data/segment-???.tfrecord \
--data_dir=data/
```
    
執行完後 data/processed 便會出現處理完的 tfrecord。
```
Waymo
├───models
├───training_configs 
├───pre-trained-models 
├───exported-models 
└───data
    ├───processed
    │   └───segment-???.tfrecord #處理後的 tfrecord
    └───segment-???.tfrecord
```

## 下載預訓練模型
到 [Tensorflow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) 下載 pretrained model。

![](https://i.imgur.com/x34zpZL.png)

我下載的是 `SSD ResNet50 V1 FPN 640x640 (RetinaNet50)`。 
* 先到 pre-trained-models 目錄下

`cd pre-trained-models`

* 下載 SSD ResNet50 的 pretrained model

```
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
```

* 解壓縮

`tar zxvf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz`

```
Waymo
├───models
├───training_configs 
├───pre-trained-models 
│   └───ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
│       ├─ checkpoint/
│       ├─ saved_model/
│       └─ pipeline.config
├───exported-models 
└───data
    ├───processed
    │   └───segment-???.tfrecord #處理後的 tfrecord
    └───segment-???.tfrecord
```

## 修改訓練用 config
到 [configs/tf2](https://github.com/tensorflow/models/tree/master/research/object_detection/configs/tf2) 找到與 pretrained model 相對應的 config，也就是ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config

* 在 training_configs 新增資料夾

```shell 
cd training_configs
mkdir ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
```

* 在 ssd_resnet50_v1_fpn_640x640_coco17_tpu-8 目錄下新增 pipeline.config，並將剛剛找到的 config 內容複製貼上，並且做一些修改。
    * num_classes: 種類個數
    * batch_size: bach size 大小，根據電腦的記憶體而有不同設置
    * fine_tune_checkpoint: 更改成 pretrained model 的 ckpt-0 路徑
    * num_steps: 訓練步數
    * use_bfloat16: 是否使用 tpu，沒有使用設定為 false
    * label_map_path: label_map.pbtxt 路徑
    * train_input_reader: 將 input_path 設定成訓練用的 tfrecord 路徑
    * metrics_set: "coco_detection_metrics"
    * use_moving_averages: false
    * eval_input_reader: 將 input_path 設定成評估用用的 tfrecord 路徑

```config
# SSD with Resnet 50 v1 FPN feature extractor, shared box predictor and focal
# loss (a.k.a Retinanet).
# See Lin et al, https://arxiv.org/abs/1708.02002
# Trained on COCO, initialized from Imagenet classification checkpoint
# Train on TPU-8
#
# Achieves 34.3 mAP on COCO17 Val

model {
  ssd {
    inplace_batchnorm_update: true
    freeze_batchnorm: false
    num_classes: 3 #因為種類有 3 個
    box_coder {
      faster_rcnn_box_coder {
        y_scale: 10.0
        x_scale: 10.0
        height_scale: 5.0
        width_scale: 5.0
      }
    }
    matcher {
      argmax_matcher {
        matched_threshold: 0.5
        unmatched_threshold: 0.5
        ignore_thresholds: false
        negatives_lower_than_unmatched: true
        force_match_for_each_row: true
        use_matmul_gather: true
      }
    }
    similarity_calculator {
      iou_similarity {
      }
    }
    encode_background_as_zeros: true
    anchor_generator {
      multiscale_anchor_generator {
        min_level: 3
        max_level: 7
        anchor_scale: 4.0
        aspect_ratios: [1.0, 2.0, 0.5]
        scales_per_octave: 2
      }
    }
    image_resizer {
      fixed_shape_resizer {
        height: 640
        width: 640
      }
    }
    box_predictor {
      weight_shared_convolutional_box_predictor {
        depth: 256
        class_prediction_bias_init: -4.6
        conv_hyperparams {
          activation: RELU_6,
          regularizer {
            l2_regularizer {
              weight: 0.0004
            }
          }
          initializer {
            random_normal_initializer {
              stddev: 0.01
              mean: 0.0
            }
          }
          batch_norm {
            scale: true,
            decay: 0.997,
            epsilon: 0.001,
          }
        }
        num_layers_before_predictor: 4
        kernel_size: 3
      }
    }
    feature_extractor {
      type: 'ssd_resnet50_v1_fpn_keras'
      fpn {
        min_level: 3
        max_level: 7
      }
      min_depth: 16
      depth_multiplier: 1.0
      conv_hyperparams {
        activation: RELU_6,
        regularizer {
          l2_regularizer {
            weight: 0.0004
          }
        }
        initializer {
          truncated_normal_initializer {
            stddev: 0.03
            mean: 0.0
          }
        }
        batch_norm {
          scale: true,
          decay: 0.997,
          epsilon: 0.001,
        }
      }
      override_base_feature_extractor_hyperparams: true
    }
    loss {
      classification_loss {
        weighted_sigmoid_focal {
          alpha: 0.25
          gamma: 2.0
        }
      }
      localization_loss {
        weighted_smooth_l1 {
        }
      }
      classification_weight: 1.0
      localization_weight: 1.0
    }
    normalize_loss_by_num_matches: true
    normalize_loc_loss_by_codesize: true
    post_processing {
      batch_non_max_suppression {
        score_threshold: 1e-8
        iou_threshold: 0.6
        max_detections_per_class: 100
        max_total_detections: 100
      }
      score_converter: SIGMOID
    }
  }
}

train_config: {
  fine_tune_checkpoint_version: V2
  #pretrained model 的 ckpt-0 位置
  fine_tune_checkpoint: "pre-trained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0" 
  fine_tune_checkpoint_type: "detection" #改為 detection
  batch_size: 2
  sync_replicas: true
  startup_delay_steps: 0
  replicas_to_aggregate: 8
  use_bfloat16: false #因為沒有使用 tpu 所以改為 false
  num_steps: 6000 #訓練步數
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }
  }
  optimizer {
    momentum_optimizer: {
      learning_rate: {
        cosine_decay_learning_rate {
          learning_rate_base: .04
          total_steps: 25000
          warmup_learning_rate: .013333
          warmup_steps: 2000
        }
      }
      momentum_optimizer_value: 0.9
    }
    use_moving_average: false
  }
  max_number_of_boxes: 100
  unpad_groundtruth_tensors: false
}

train_input_reader: {
  label_map_path: "./label_map.pbtxt"
  tf_record_input_reader {
    input_path: "data/processed/*.tfrecord"
  }
}

eval_config: {
  metrics_set: "coco_detection_metrics"
  use_moving_averages: false
}

eval_input_reader: {
  label_map_path: "./label_map.pbtxt"
  shuffle: false
  num_epochs: 1
  tf_record_input_reader {
    input_path: "data/processed/*.tfrecord"
  }
}
```
```
Waymo
├───models
├───training_configs 
│   └───ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
│       └───pipeline.config #新增 pipeline.config
├───pre-trained-models 
│   └───ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
│       ├─ checkpoint/
│       ├─ saved_model/
│       └─ pipeline.config
├───exported-models 
└───data
    ├───processed
    │   └───segment-???.tfrecord
    └───segment-???.tfrecord
```
## 訓練模型

**model_main_tf2.py**

model_dir: 會將訓練的 checkpoint 儲存在 model_dir 目錄下

pipeline_config_path: pipeline.config 路徑

執行方式如下: 

```shell 
python model_main_tf2.py \
--model_dir=training_configs/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8 \
--pipeline_config_path=training_configs/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/pipeline.config
```
執行結果如下: 每 100 steps 會印一次。
```
Step 2100 per-step time 0.320s
INFO:tensorflow:{'Loss/classification_loss': 0.121629156,
 'Loss/localization_loss': 0.16370133,
 'Loss/regularization_loss': 0.2080817,
 'Loss/total_loss': 0.4934122,
 'learning_rate': 0.039998136}
I0605 08:29:04.605577 139701982308224 model_lib_v2.py:700] {'Loss/classification_loss': 0.121629156,
 'Loss/localization_loss': 0.16370133,
 'Loss/regularization_loss': 0.2080817,
 'Loss/total_loss': 0.4934122,
 'learning_rate': 0.039998136}
```

## 評估模型 (Optional)
**model_main_tf2.py**

checkpoint_dir: 讀取 checkpoint 的目錄。

執行方式如下: 

```shell 
python model_main_tf2.py \
--model_dir=training_configs/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8 \
--pipeline_config_path=training_configs/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/pipeline.config \
--checkpoint_dir=training_configs/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/ 
```
執行結果: 會計算 AP 和 AR

$AP^{small}:$ AP for small object : area < $32^2$

$AP^{medium}:$ AP for medium object : $32^2$ < area < $96^2$ 

$AP^{large}:$ AP for large object : $96^2$ < area

![](https://i.imgur.com/RjN2dRf.png)

## 輸出模型
**exporter_main_v2.py**

input_type: image_tensor

pipeline_config_path:  pipeline.config 的路徑

trained_checkpoint_dir: 儲存 checkpoint 的位置

output_directory: 輸出模型位置

執行方式如下: 

```shell 
!python exporter_main_v2.py \
--input_type image_tensor \
--pipeline_config_path training_configs/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/pipeline.config \
--trained_checkpoint_dir training_configs/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/  \
--output_directory exported-models/my_model_6000steps
```
執行結果如下:
```shell 
INFO:tensorflow:Assets written to: exported-models/my_model_6000steps/saved_model/assets
I0605 09:07:21.034602 139745385867136 builder_impl.py:775] Assets written to: exported-models/my_model_6000steps/saved_model/assets
INFO:tensorflow:Writing pipeline config file to exported-models/my_model_6000steps/pipeline.config
I0605 09:07:22.310333 139745385867136 config_util.py:254] Writing pipeline config file to exported-models/my_model_6000steps/pipeline.config
```
```
Waymo
├───models
├───training_configs 
│   └───ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
│       └─pipeline.config #新增 pipeline.config
├───pre-trained-models 
│   └───ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
│       ├─ checkpoint/
│       ├─ saved_model/
│       └─ pipeline.config
├───exported-models 
│   └───my_model_6000steps
└───data
    ├───processed
    │   └─segment-???.tfrecord
    └───segment-???.tfrecord
```
## 使用模型預測圖片
**detect.py**

saved_model_path: 模型位置

test_path: 測試圖片位置

output_path: 輸出預測圖片位置

min_score_thresh: 信心水準

執行方式如下:

```shell 
!python detect.py \
--saved_model_path=exported-models/my_model_6000steps \
--test_path=test_image \
--output_path=output_image \
--min_score_thresh=.1
```
預測結果:

<img src="https://i.imgur.com/NNE6OuI.png" width=250px height=200px> 
<img src="https://i.imgur.com/dyRuUpA.png" width=300px height=200px>
<img src="https://i.imgur.com/vICSrnI.png" width=250px height=200px>
<img src="https://i.imgur.com/it53kPf.png" width=300px height=200px>
