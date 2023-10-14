from pathlib import Path
from PIL import Image
import numpy as np
from typing import Tuple

dataset_path = Path("./dataset")


def resize_image(image: Image) -> Image:
    return image.resize((46, 56))


def get_image_array(image: Image) -> np.ndarray:
    image_array: list[int] = []
    for y in range(image.height):
        for x in range(image.width):
            image_array.append(image.getpixel((x, y)))
    return np.array(image_array)


def get_mean_center_array(image_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean_array: np.ndarray = np.mean(image_array, axis=0)
    tiled_mean_array: np.ndarray = np.tile(mean_array, (image_array.shape[0], 1))
    return image_array - tiled_mean_array, mean_array


def convert_array_to_image(image_array: np.ndarray) -> Image:
    image_array = image_array.reshape((56, 46))
    image_array = image_array
    image_array = image_array.astype(np.uint8)
    return Image.fromarray(image_array)


def get_covariance_matrix(input_matrix: np.ndarray) -> np.ndarray:
    return np.cov(input_matrix, rowvar=False)


def get_train_dataset() -> Tuple[np.ndarray, list[str]]:
    output_matrix = []
    labels = []
    for subset in dataset_path.glob('*/'):
        for numbered_image in subset.glob('*.pgm'):
            if int(numbered_image.stem) <= 5:
                output_matrix.append(get_image_array(resize_image(Image.open(numbered_image))))
                labels.append(subset.name)
    return np.array(output_matrix), labels


original_train_dataset, labels = get_train_dataset()

mean_matrix, mean_array = get_mean_center_array(original_train_dataset)  # get mean matrix

covariance_matrix = get_covariance_matrix(mean_matrix)  # get covariance matrix

eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

# eigen_values = np.abs(eigen_values)
#
# indices = np.argsort(eigen_values)
#
# indices = indices[:400]

filtered_eigen_vectors = eigen_vectors[:, :200]

# convert_array_to_image(filtered_eigen_vectors[:, 32] * 100).show()

projection_coefficients = mean_matrix @ filtered_eigen_vectors

test_input = dataset_path / 's1' / '9.pgm'


def get_coefficients_by_image(image: Image) -> np.ndarray:
    image_array = get_image_array(image)
    image_array = image_array - mean_array
    return image_array @ filtered_eigen_vectors


def get_coefficients_distance(input_coefficients: np.ndarray) -> np.ndarray:
    tiled_coefficients = np.tile(input_coefficients, (projection_coefficients.shape[0], 1))
    distance = np.sqrt(np.sum((projection_coefficients - tiled_coefficients) ** 2, axis=1))
    return distance


def test():
    error_count = 0
    for subset in dataset_path.glob('*/'):
        for file in subset.glob('*.pgm'):
            if int(file.stem) <= 5:
                continue
            test_coefficients = get_coefficients_by_image(resize_image(Image.open(file)))
            result = get_coefficients_distance(test_coefficients)
            sorted_result = np.argsort(result)
            result_label = labels[sorted_result[0]]
            if result_label == subset.name:
                print(f"Test {subset.name} - {file.stem} success")
            else:
                print(f"Test {subset.name} - {file.stem} failed")
                error_count += 1
    print(f"Total error count: {error_count}. Error rate: {error_count / 200 * 100}%")

test()

