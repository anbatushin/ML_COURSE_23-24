import numpy as np
from typing import Tuple


def sum_non_neg_diag(X: np.ndarray) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    s = np.diag(X)
    if any(s >= 0):
        return s[s >= 0].sum()
    return -1


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    return (np.sort(x) == np.sort(y)).all()


def max_prod_mod_3(x: np.ndarray) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    mas = x[:-1]
    mas_2 = x[1:]
    s = mas * mas_2
    if len(s[s % 3 == 0]) == 0:
        return -1
    return max(s[s % 3 == 0])


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Сложить каналы изображения с указанными весами.
    """
    new_img = np.add.reduce(image[:] * weights, axis=-1)
    return new_img.reshape(image.shape[0], image.shape[1])


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    x2 = np.repeat(np.transpose(x)[0], np.transpose(x)[1])
    y2 = np.repeat(np.transpose(y)[0], np.transpose(y)[1])
    if len(x2) != len(y2):
        return -1
    else:
        return np.dot(x2, y2)


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y.
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    matrix = np.matmul(X, Y.T)
    matrix_1 = np.linalg.norm(X, axis=-1).reshape(X.shape[0], 1)
    matrix_2 = np.linalg.norm(Y, axis=-1).reshape(Y.shape[0], 1)
    m_2 = np.matmul(matrix_1, matrix_2.T)
    flag = m_2 != 0
    m_1 = np.divide(matrix, m_2, where=flag)
    np.putmask(m_1, m_2 == 0.0, 1.0)
    return m_1
