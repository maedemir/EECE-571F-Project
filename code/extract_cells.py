import os
import json
import PIL.Image as Image


IMAGE_PATH = 'data/imgs'
CELL_IMAGE_PATCHES_DIR = 'data/extracted_cells'
FEATURES_OUTPUT_DIR = 'data/nuclei_features'
CLASSES = ['HP', 'NCM', 'SSL', 'TA']
JSON_DIR_PATH = 'data/json'


def open_json(json_path):
    with open(json_path) as f:
        return json.load(f)


def get_bboxes():
    bbox_per_file = {}
    for cls in CLASSES:
        cls_json_dir_path = os.path.join(JSON_DIR_PATH, cls, 'json')
        for json_file_name in os.listdir(cls_json_dir_path):
            json_path = os.path.join(cls_json_dir_path, json_file_name)
            json_content = open_json(json_path)

            bbox_values = {}
            for key, value in json_content.get("nuc", {}).items():
                bbox = value.get("bbox")
                if bbox is not None:
                    bbox_values[key] = bbox

            bbox_per_file[json_file_name] = bbox_values
    return bbox_per_file


def get_class_name(file_name):
    class_name = file_name.split('-')[-1].split('_')[0]
    return class_name


def extract_cells(bboxes, image_name, image_class):
    output_path = os.path.join(CELL_IMAGE_PATCHES_DIR, image_class, image_name)
    os.makedirs(output_path, exist_ok=True)

    image_path = os.path.join(IMAGE_PATH, image_class, image_name)
    img = Image.open(image_path + '.png')

    for node_name, ((x1, y1), (x2, y2)) in bboxes.items():
        # Extract patch from the image
        patch = img.crop((y1, x1, y2, x2))

        # Save the patch to the output folder
        patch_path = os.path.join(
            output_path, f'{image_name}-patch_{node_name.zfill(4)}.png')
        patch.save(patch_path)


if __name__ == '__main__':
    os.makedirs(CELL_IMAGE_PATCHES_DIR, exist_ok=True)
    os.makedirs(FEATURES_OUTPUT_DIR, exist_ok=True)

    bboxes_per_file = get_bboxes()
    for file_name, bboxes in bboxes_per_file.items():
        image_name = file_name.split('/')[-1].split('.')[0]
        image_class = get_class_name(image_name)
        extract_cells(bboxes=bboxes, image_name=image_name,
                      image_class=image_class)
