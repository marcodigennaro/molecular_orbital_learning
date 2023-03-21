import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torchvision.io import read_image
import cv2
from proj_paths import ROOT_DIR, FIGS_DIR

if __name__ == "__main__":

    df = pd.read_csv('distorted_orbitals.csv', index_col=0)
    train_size = int(len(df) * 0.8)
    test_size = len(df) - train_size
    print(f'Length df = {len(df)}\n       train = {train_size}\n       test = {test_size}')

    # Have a look at the dataset
    sample = df.sample()
    img_path = Path(FIGS_DIR).joinpath(sample['fig_name'].values[0])

    # visualize image with opencv
    img = cv2.imread(str(img_path))
    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY ))
    #plt.show()

    # visualize image with PIL
    img1 = Image.open(img_path)


    # visualize image with PT
    # https://www.tutorialspoint.com/how-to-read-a-jpeg-or-png-image-in-pytorch
    img2 = read_image(str(img_path))
    img2 = T.ToPILImage()(img2)
    #img2.show()


    for check_img in [img1, img2]:
        # check if input image is a PyTorch tensor
        print("Is image a PyTorch Tensor:", torch.is_tensor(check_img))
        print("Type of Image:", type(check_img))
        # size of the image
        print(check_img.size)
        image_width, image_height = T.functional.get_image_size(check_img)  # 480x480
        print(image_width, image_height)
        # display the image properties
        print("Image data:", check_img)


        # convert PIL image to numpy array
        img_np = np.array(check_img)
        # plot the pixel values
        plt.hist(img_np.ravel(), bins=50, density=True)
        plt.xlabel("pixel values")
        plt.ylabel("relative frequency")
        plt.title("distribution of pixels")
        plt.show()