import fire

import numpy as np

from pathlib import Path
from PIL import Image


def main(
    npz_path:str,
    out_dir:str,
):
    data = np.load(npz_path)['arr_0']
    out_dir = Path(out_dir)
    for i in range(data.shape[0]):
        img = Image.fromarray(data[i])
        img.save(out_dir / f'{i}.jpg')


if __name__ == '__main__':
    fire.Fire(main)