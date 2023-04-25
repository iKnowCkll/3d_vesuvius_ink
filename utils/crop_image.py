import cv2
import gc
import numpy as np
import os
import pandas as pd
import PIL.Image as Image
import tifffile as tiff
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm.notebook import tqdm

OUTPUT_PNG = True
OUTPUT_NPY = False

ROI_SIZE = 512
IMAGE_NUM = 3

INPUT_DIR = Path("E:/kaggle/vesuvius-ink-detection/train")
OUTPUT_DIR = Path("E:/kaggle/3d_vesuvius_ink_detection/512_data")


def crop_image(image_id_, input_dir, output_dir):
    output_data_dir = Path(output_dir / f"{image_id_}")
    if not output_data_dir.exists():
        output_data_dir.mkdir()

    # Load label image
    inklabels_img = np.array(Image.open(str(input_dir / f"{image_id_}" / "inklabels.png")))
    inklabels_img = np.pad(inklabels_img, [(0, ROI_SIZE - inklabels_img.shape[0] % ROI_SIZE),
                                           (0, ROI_SIZE - inklabels_img.shape[1] % ROI_SIZE)], 'constant')

    # Get the scanning position of the image
    x_pos_list = []
    y_pos_list = []
    for y in range(0, inklabels_img.shape[0], ROI_SIZE):
        for x in range(0, inklabels_img.shape[1], ROI_SIZE):
            if inklabels_img[y:y + ROI_SIZE, x:x + ROI_SIZE].max() > 0:
                x_pos_list.append(x)
                y_pos_list.append(y)

    # Crop the image
    image_path_list = sorted(list(Path(input_dir / f"{image_id_}" / "surface_volume").glob('*.tif')))
    for i, image_path in tqdm(enumerate(image_path_list), total=len(image_path_list),
                              desc=f"Cropping images - {image_id_}", dynamic_ncols=True):

        # load image
        img = tiff.imread(str(image_path))
        img = np.pad(img, [(0, ROI_SIZE - img.shape[0] % ROI_SIZE), (0, ROI_SIZE - img.shape[1] % ROI_SIZE)],
                     'constant')

        # crop
        for j, (x, y) in enumerate(zip(x_pos_list, y_pos_list)):

            image_roi = img[y:y + ROI_SIZE, x:x + ROI_SIZE]
            image_roi = image_roi.astype(np.float32) / 65535.0

            if OUTPUT_NPY:
                np.save(str(output_data_dir / f"{j:03d}_{i:02d}"), image_roi)

            if OUTPUT_PNG:
                output_image_dir = Path(output_data_dir / f"{j:03d}")
                if not output_image_dir.exists():
                    output_image_dir.mkdir()
                cv2.imwrite(str(output_image_dir / f"{i:02d}.png"), (image_roi * 255).astype(np.uint8))

        del img
        gc.collect()

    # Create training data
    image_id_list = []
    roi_id_list = []
    for j, (x, y) in tqdm(enumerate(zip(x_pos_list, y_pos_list)), total=len(x_pos_list),
                          desc=f"Merge images - {image_id_}", dynamic_ncols=True):

        # input
        if OUTPUT_NPY:
            image_tiles = []
            for i in range(len(image_path_list)):
                filename = str(output_data_dir / f"{j:03d}_{i:02d}.npy")
                image_tiles.append(np.load(filename))
                os.remove(filename)
            np.save(str(output_data_dir / f"{j:03d}"), np.stack(image_tiles, axis=0))

        # mask
        image_roi = inklabels_img[y:y + ROI_SIZE, x:x + ROI_SIZE]
        cv2.imwrite(str(output_data_dir / f"{j:03d}.png"), (image_roi * 255).astype(np.uint8))

        # metadata
        image_id_list.append(image_id_)
        roi_id_list.append(f"{j:03d}")

    # Drawing to check the cropped position
    inklabels_img = (inklabels_img * 255).astype(np.uint8)
    inklabels_img = cv2.cvtColor(inklabels_img, cv2.COLOR_GRAY2BGR)
    for x, y in zip(x_pos_list, y_pos_list):
        inklabels_img = cv2.rectangle(
            inklabels_img,
            (x, y),
            (x + ROI_SIZE, y + ROI_SIZE),
            (0, 255, 0),
            thickness=10)

    cv2.imwrite(str(output_dir / f"crop_image{image_id_}.png"), inklabels_img)
    plt.imshow(cv2.cvtColor(inklabels_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    return {
        "image_id_list": image_id_list,
        "roi_id_list": roi_id_list,
        "x_pos_list": x_pos_list,
        "y_pos_list": y_pos_list,
    }


image_id_list_all = []
roi_id_list_all = []
x_pos_list_all = []
y_pos_list_all = []
for image_id in range(1, IMAGE_NUM + 1):
    outputs = crop_image(image_id, INPUT_DIR, OUTPUT_DIR)

    image_id_list_all.extend(outputs["image_id_list"])
    roi_id_list_all.extend(outputs["roi_id_list"])
    x_pos_list_all.extend(outputs["x_pos_list"])
    y_pos_list_all.extend(outputs["y_pos_list"])

df = pd.DataFrame({
    "image_id": image_id_list_all,
    "roi_id": roi_id_list_all,
    "x_pos": x_pos_list_all,
    "y_pos": y_pos_list_all,
})
df.to_csv(str(OUTPUT_DIR / "train.csv"), index=False)
