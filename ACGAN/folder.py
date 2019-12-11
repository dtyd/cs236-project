import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from PIL import Image
import os
import os.path
import numpy as np
import pdb

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir, classes_idx=None):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    if classes_idx is not None:
        assert type(classes_idx) == tuple
        start, end = classes_idx
        classes = classes[start:end]
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        if target not in class_to_idx:
            continue
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class CaptionedImageDataset(Dataset):
    def __init__(self, image_shape, n_classes):
        self.image_shape = image_shape
        self.n_classes = n_classes

    def __getitem__(self, index: int) -> (torch.tensor, torch.tensor, list):
        '''
        :param index: index of the element to be fetched
        :return: (image : torch.tensor , class_ids : torch.tensor ,captions : list(str))
        '''
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class Imagenet32Dataset(CaptionedImageDataset):
    def __init__(self, root="datasets/ImageNet32", train=True, max_size=-1):
        '''
        :param dirname: str, root dir where the dataset is downloaded
        :param train: bool, true if train set else val
        :param max_size: int, truncate size of the dataset, useful for debugging
        '''
        super().__init__((3, 32, 32), 1000)
        self.root = root
        if train:
            self.dirname = os.path.join(root, "train")
        else:
            self.dirname = os.path.join(root, "val")

        # self.classId2className = load_vocab_imagenet(os.path.join(root, "map_clsloc.txt"))
        self.classId2className = {60: 'brown bear', 258: 'shopping cart', 366: 'seashore', 428: 'crane', 499: 'tree frog', 567: 'carousel', 670: 'frying pan', 705: 'bookshop', 907: 'basketball', 992: 'cheeseburger'}
        classes = [60,258,366,428,499,567,670,705, 907,992]
        data_files = sorted(os.listdir(self.dirname))
        all_images = []
        all_labelIds = []
        for i, f in enumerate(data_files):
            print("loading data file {}/{}, {}".format(i + 1, len(data_files), os.path.join(self.dirname, f)))
            data = np.load(os.path.join(self.dirname, f))
            all_images.append(data['data'])
            all_labelIds.append(data['labels'] - 1)
        images = np.concatenate(all_images, axis=0)
        labelIds = np.concatenate(all_labelIds)

        small_images = []
        small_labelIds = []
        n = len(labelIds)
        for i in range(n):
            if labelIds[i] in self.classId2className:
                small_images.append(images[i])
                new_class = int(np.where(classes == labelIds[i])[0])
                small_labelIds.append(new_class)

        self.images = np.vstack(small_images)
        self.labelIds = np.array(small_labelIds)

        if max_size >= 0:
            # limit the size of the dataset
            self.labelNames = self.labelNames[:max_size]
            self.labelIds = self.labelIds[:max_size]


    def __getitem__(self, index: int) -> (torch.tensor, torch.tensor, list):
        image = torch.tensor(self.images[index]).reshape(3, 32, 32).float() / 128 - 1
        label = self.labelIds[index]
        #caption = self.labelNames[index].replace("_", " ")
        #return (image, label, caption)
        return (image, label)

    def __len__(self):
        return len(self.labelIds)


class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, classes_idx=None):
        self.classes_idx = classes_idx
        classes, class_to_idx = find_classes(root, self.classes_idx)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
