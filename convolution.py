# USAGE
# python convolution.py --image panda.jpg

from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2


def convolve(image, kernel):
    # Получаем размеры изображения и маски
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    # Определяем размер отступа по маске
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")

    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # Интересующая нас область из изображения с оступами
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            # Convolution
            k = (roi * kernel).sum()

            # Забираем область изображения без рамок
            output[y - pad, x - pad] = k

    # Устанавливем интенсивность в диапазоне [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")
    return output

# Парсинг аргументов
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

# Маски для блюра
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))

# Улучшение четкости
sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")

kernelBank = (
    ("small_blur", smallBlur),
    ("sharpen", sharpen),
)

# Загрузка изображения и преобразование
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

for (kernelName, kernel) in kernelBank:
    # Применение нашего фильтра и фильтра библиотеки OpenCV
    print("[INFO] applying {} kernel".format(kernelName))
    convoleOutput = convolve(gray, kernel)
    opencvOutput = cv2.filter2D(gray, -1, kernel)

    # Вывод изображений
    cv2.imshow("original", gray)
    cv2.imshow("{} - convole".format(kernelName), convoleOutput)
    cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()