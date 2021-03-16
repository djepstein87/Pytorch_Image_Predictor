from image_classifier import imagenet_class_index, my_model, parse


def train():
    pass


def predict(image_bytes):
    tensor = parse.transform_image(image_bytes=image_bytes)

    # uses pre-loaded model (loaded at init) to make prediction
    predicted_idx = get_predicted_index(my_model, tensor)

    # get label from pre-loaded imagenet file
    image_class = imagenet_class_index[predicted_idx]

    return image_class


def get_predicted_index(model, tensor):
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)

    predicted_idx = str(y_hat.item())

    return predicted_idx
