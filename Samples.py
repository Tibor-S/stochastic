from email.mime import image
import random
import mnist
import numpy as np
import matplotlib.pyplot as plt


class DigitSample:
    bitmap: list[list[int]]
    label: int

    def __init__(self, test=False, index=0, randomD=False):
        if test:
            images, labels = mnist.test_images(), mnist.test_labels()
        else:
            images, labels = mnist.train_images(), mnist.train_labels()
        if randomD:
            index = random.randrange(0, len(images))  # choose an index ;-)
        self.bitmap = images[index]
        self.label = labels[index]


if __name__ == '__main__':
    DigitSample(test=True, randomD=True)
