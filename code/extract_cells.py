import os
import json
import argparse
import PIL.Image as Image


CLASSES = ['HP', 'NCM', 'SSL', 'TA']


def open_json(json_path):
    with open(json_path) as f:
        return json.load(f)


def get_bboxes(args):
    bbox_per_file = {}
    cls_json_dir_path = args.json_dir
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


def extract_cells(args, bboxes, image_name, image_class):
    output_path = os.path.join(args.cell_image_patches_dir, image_class, image_name)
    os.makedirs(output_path, exist_ok=True)

    image_path = os.path.join(args.image_dir, image_class, image_name)
    img = Image.open(image_path + '.png')

    for node_name, ((x1, y1), (x2, y2)) in bboxes.items():
        # Extract patch from the image
        patch = img.crop((y1, x1, y2, x2))

        # Save the patch to the output folder
        print(os.path.join(output_path, f'{image_name}-patch_{node_name.zfill(4)}.png'))
        patch_path = os.path.join(output_path, f'{image_name}-patch_{node_name.zfill(4)}.png')
        patch.save(patch_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract cells based on hovernet json outputs.")

    parser.add_argument("--image_dir", default='')
    parser.add_argument("--cell_image_patches_dir", default='')
    parser.add_argument("--json_dir", default='')

    args, unknown = parser.parse_known_args()

    os.makedirs(args.cell_image_patches_dir, exist_ok=True)

    image_class = args.cell_image_patches_dir.split('/')[-2]

    bboxes_per_file = get_bboxes(args)
    for file_name, bboxes in bboxes_per_file.items():
        image_name = file_name.split('/')[-1].split('.')[0]
        # image_class = get_class_name(image_name)
        extract_cells(args, bboxes=bboxes, image_name=image_name,
                      image_class=image_class)
