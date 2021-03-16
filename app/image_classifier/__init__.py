import json

from torchvision import models, transforms


def load_model():
    model = models.densenet121(pretrained=True)
    model.eval()

    return model


my_model = load_model()

my_transforms = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        [0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])])

imagenet_class_index = json.load(open('image_classifier/data/imagenet_class_index.json'))
