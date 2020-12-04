import time
import os
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
from tensorflow import keras
from PIL import Image


flags.DEFINE_string('classes', './data/yolo.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_string('data_dir', '', 'path to classifier dataset')
flags.DEFINE_string('model', "model", 'path to model directory')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 832, 'resize images to')
flags.DEFINE_integer('imgsize', 224, 'size of model images')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 2, 'number of classes in the model')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    model = keras.models.load_model(FLAGS.model)

    img_raw = tf.image.decode_image(
        open(FLAGS.image, 'rb').read(), channels=3)

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, FLAGS.size)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(FLAGS.data_dir, validation_split=0.2, seed=69,
        subset="training", image_size=(FLAGS.imgsize, FLAGS.imgsize), batch_size=32, interpolation="lanczos5")

    plane_classes = [[0]*int(nums[0])]
    plane_scores = [[0]*int(nums[0])]
    outputs = [[""]*int(nums[0])]

    for i in range(nums[0]):
        imgc = Image.open(FLAGS.image)
        width, height = imgc.size
        bounds = np.array(boxes[0][i])
        bounds[0] = bounds[0] * width
        bounds[1] = bounds[1] * height
        bounds[2] = bounds[2] * width
        bounds[3] = bounds[3] * height
        imgc.crop(bounds).save(os.path.join("yoloCrop", str(i) + ".jpg"))
     
        imgc = keras.preprocessing.image.load_img(
            "yoloCrop/" + str(i) + ".jpg", target_size=(FLAGS.imgsize, FLAGS.imgsize)) 
    
        img_array = keras.preprocessing.image.img_to_array(imgc)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
    
        plane_classes[0][i] = np.argmax(score)
        plane_scores[0][i] = 100 * np.max(score)
    
        logging.info('\t{}, {}, {}'.format(train_ds.class_names[int(plane_classes[0][i])],
                                           np.array(plane_scores[0][i] / 100),
                                           np.array(bounds)))
        outputs[0][i] = str(np.array(scores[0][i])) + " " + str(np.array(plane_scores[0][i] / 100))

    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, outputs, plane_classes, nums), train_ds.class_names)
    cv2.imwrite(FLAGS.output, img)
    logging.info('output saved to: {}'.format(FLAGS.output))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
