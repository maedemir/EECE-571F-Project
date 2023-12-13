import os
import json
import argparse
import PIL.Image as Image

from utils import get_bboxes

CLASSES = ['HP', 'NCM', 'SSL', 'TA']


def extract_cells(args, bboxes, image_name, image_class):
    output_path = os.path.join(args.cell_image_patches_dir, image_name)
    os.makedirs(output_path, exist_ok=True)

    image_path = os.path.join(args.image_dir, image_class, image_name)
    img = Image.open(image_path + '.png')

    for node_name, ((x1, y1), (x2, y2)) in bboxes.items():
        # Extract patch from the image
        patch = img.crop((y1, x1, y2, x2))

        # Save the patch to the output folder
        patch_path = os.path.join(
            output_path, f'{image_name}-patch_{node_name.zfill(4)}.png')
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
