import torchvision.transforms as transforms


def basic_transform():
    transform_list = []
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5,), (0.5,))]

    return transforms.Compose(transform_list)


def normalize_cxr(image):
    return image / 4095
