import time
import os
import hashlib

from absl import app, flags, logging
from absl.flags import FLAGS
import lxml.etree
import tqdm
from PIL import Image

flags.DEFINE_string('data_dir', '',
                    'path to raw PASCAL VOC dataset')
flags.DEFINE_string('output_dir', '', 'outpot dataset')
flags.DEFINE_enum('split', 'train', [
                  'train', 'val'], 'specify train or val spit')

def create_img(xmin, ymin, xmax, ymax, filename, count, classtxt):
    if not os.path.exists(os.path.join(FLAGS.output_dir, classtxt)):
        os.makedirs(os.path.join(FLAGS.output_dir, classtxt))

    img = Image.open(os.path.join(FLAGS.data_dir, 'JPEGImages', filename))
    width, height = img.size
    if(xmin < 0):
        xmin = 0
    if(ymin < 0):
        ymin = 0
    if(xmax > width):
        xmax = width - 1
    if(ymax > height):
        ymax = height - 1
    filename = filename.replace(".jpg", "_"+str(count)+".jpg")
    img.crop((xmin, ymin, xmax, ymax)).save(os.path.join(FLAGS.output_dir, classtxt, filename))


def parse_img(annotation):
    img_path = os.path.join(
        FLAGS.data_dir, 'JPEGImages', annotation['filename'])
    img_raw = open(img_path, 'rb').read()
    key = hashlib.sha256(img_raw).hexdigest()

    width = int(annotation['size']['width'])
    height = int(annotation['size']['height'])

    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0
    classes_text = ""
    if 'object' in annotation:
        filename = "" + annotation['filename']
        i = 0
        for obj in annotation['object']:
            xmin = int(obj['bndbox']['xmin'])
            ymin = int(obj['bndbox']['ymin'])
            xmax = int(obj['bndbox']['xmax'])
            ymax = int(obj['bndbox']['ymax'])
            classes_text = "" + obj['name']
            create_img(xmin, ymin, xmax, ymax, filename, i, classes_text)
            i = i + 1

def parse_xml(xml):
    if not len(xml):
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = parse_xml(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}

def main(_argv):
    image_list = open(os.path.join(
        FLAGS.data_dir, 'ImageSets', 'Main', '%s.txt' % FLAGS.split)).read().splitlines()
    logging.info("Image list loaded: %d", len(image_list))
    for name in tqdm.tqdm(image_list):
        annotation_xml = os.path.join(
            FLAGS.data_dir, 'Annotations', name + '.xml')
        annotation_xml = lxml.etree.fromstring(open(annotation_xml).read())
        annotation = parse_xml(annotation_xml)['annotation']
        parse_img(annotation)
    logging.info("Done")

if __name__ == '__main__':
    app.run(main)