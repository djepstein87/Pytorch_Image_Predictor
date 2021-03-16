import io

from PIL import Image

from image_classifier import my_transforms


def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))

    transformed_image = my_transforms(image)

    final_image = transformed_image.unsqueeze(0)

    return final_image
