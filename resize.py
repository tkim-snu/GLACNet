import argparse
import os
from PIL import Image


def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.ANTIALIAS)

def resize_images(image_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    for i, image in enumerate(images):
        image_path = os.path.join(image_dir, image)
        with open(image_path, 'r+b') as f:
            try:
                with Image.open(f) as img:
                    img = resize_image(img, size)
                    img.save(os.path.join(output_dir, image), img.format)
            except Exception:
                print("image open error : " + str(image_path))

        if i % 100 == 0:
            print ("[%d/%d] Resized the images and saved into '%s'."
                   %(i, num_images, output_dir))

def main(args):
    splits = ['train', 'val']
    for split in splits:
        image_dir = args.image_dir
        output_dir = args.output_dir
        image_size = [args.image_size, args.image_size]
        resize_images(image_dir, output_dir, image_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./data/images/train/',
                        help='directory for train images')
    parser.add_argument('--output_dir', type=str, default='./data/resized/images/train/',
                        help='directory for saving resized images')
    parser.add_argument('--image_size', type=int, default=256,
                        help='size for image after processing')
    args = parser.parse_args()
    main(args)
