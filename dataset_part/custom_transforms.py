#!/usr/bin/python
# -*- encoding: utf-8 -*-


from PIL import Image
import PIL.ImageEnhance as ImageEnhance
import random
import numpy as np


class custom_RandomCrop(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, image_label):
        image = image_label['image']
        label = image_label['label']
        assert image.size == label.size
        W, H = self.size
        w, h = image.size

        if (W, H) == (w, h): return {'image': image, 'label': label}
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            image = image.resize((w, h), Image.BILINEAR)
            label = label.resize((w, h), Image.NEAREST)
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        return dict(
                image = image.crop(crop),
                label = label.crop(crop)
                    )


class custom_HorizontalFlip(object):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = p

    def __call__(self, image_label):
        if random.random() > self.p:
            return image_label
        else:
            image = image_label['image']
            label = image_label['label']
            return {'image': image.transpose(Image.FLIP_LEFT_RIGHT), 'label': label.transpose(Image.FLIP_LEFT_RIGHT)}


class custom_RandomScale(object):
    def __init__(self, scales=(1, ), *args, **kwargs):
        self.scales = scales
        # print('scales: ', scales)

    def __call__(self, image_label):
        image = image_label['image']
        label = image_label['label']
        W, H = image.size
        scale = random.choice(self.scales)
        # scale = np.random.uniform(min(self.scales), max(self.scales))
        w, h = int(W * scale), int(H * scale)
        return {'image': image.resize((w, h), Image.BILINEAR), 'label': label.resize((w, h), Image.NEAREST)}


class custom_ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if not brightness is None and brightness>0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast>0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation>0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, image_label):
        image = image_label['image']
        label = image_label['label']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        image = ImageEnhance.Brightness(image).enhance(r_brightness)
        image = ImageEnhance.Contrast(image).enhance(r_contrast)
        image = ImageEnhance.Color(image).enhance(r_saturation)
        return {'image': image, 'label': label}

class custom_MultiScale(object):
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, image):
        W, H = image.size
        sizes = [(int(W*ratio), int(H*ratio)) for ratio in self.scales]
        images = []
        [images.append(image.resize(size, Image.BILINEAR)) for size in sizes]
        return images


class custom_Compose(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, image_label):
        for comp in self.do_list:
            image_label = comp(image_label)
        return image_label