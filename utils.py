import numbers
import torch
import PIL.Image as Image
from torchvision.transforms import v2


def loadRGB(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def get_padding(image):
    w, h = image.size
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding


class NewPad(object):
    def __init__(self, fill=0, padding_mode="constant"):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ["constant", "edge", "reflect", "symmetric"]

        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return v2.functional.pad(
            img, list(get_padding(img)), self.fill, self.padding_mode
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(padding={0}, fill={1}, padding_mode={2})".format(
                self.fill, self.padding_mode
            )
        )

class MyLazyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.dataset[index][0])
        else:
            x = self.dataset[index][0]
        y = self.dataset[index][1]
        return x, y
    
    def __len__(self):
        return len(dataset)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, the_dataset, the_indices, the_transform=None):
        self.dataset = the_dataset
        self.transform = the_transform
        self.indices = the_indices
        
    def __getitem__(self, index):
        idx = self.indices[index]
        if self.transform:
            x = self.transform(self.dataset[idx][0])
        else:
            x = self.dataset[idx][0]
        y = self.dataset[idx][1]
        return x, y
    
    def __len__(self):
        return len(self.indices)

trans = tv.transforms.Compose(
    [
        NewPad(padding_mode="constant"),
        tv.transforms.Resize(256),
        tv.transforms.RandomRotation(5),
        tv.transforms.RandomPerspective(distortion_scale=0.15, p=1.0),
        tv.transforms.RandomResizedCrop(224, scale=(.5, 1.0)),
        # tv.transforms.CenterCrop(224),
        # tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

transNoAugment = tv.transforms.Compose(
    [
        NewPad(padding_mode="constant"),
        tv.transforms.Resize(224),
        # tv.transforms.Resize(256),
        # tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
