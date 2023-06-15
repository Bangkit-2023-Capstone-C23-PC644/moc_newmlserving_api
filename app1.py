from fastapi import FastAPI, APIRouter, File, UploadFile
import numpy as np
import tensorflow as tf
import cv2
import numpy as np
import uvicorn
import os
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.builders import model_builder

app = FastAPI()

configs = config_util.get_configs_from_pipeline_file('./models_trained/my_ssd_mobnet/pipeline.config')
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join('./models_trained/my_ssd_mobnet', 'ckpt-3')).expect_partial()

def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap('./label_map.pbtxt')

def testing(img_path, threshold):
  img = cv2.imread(img_path)
  image_np = np.array(img)

  input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
  #input_tensor = tf.convert_to_tensor(np.expand_dims(img, 0), dtype=tf.float32)
  #input_tensor = tf.image.decode_png(tf.io.read_file(IMAGE_PATH), channels=3)[tf.newaxis, ...]

  detections = detect_fn(input_tensor)

  num_detections = int(detections.pop('num_detections'))
  detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
  detections['num_detections'] = num_detections

  # detection_classes should be ints.
  detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

  detect_score = detections['detection_scores']
  return detect_score

def count_person(score):
  terdeteksi = []
  for i in score:
    if i > 0.5:
      terdeteksi.append(i)
  return len(terdeteksi)



@app.get("/ping")
async def ping():
    return "hello"


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    contents = await file.read()
    with open(file.filename, "wb") as f:
        f.write(contents)

    x = count_person(testing(file.filename,0.5))
    os.remove(file.filename)
    return {"estimate": x}
