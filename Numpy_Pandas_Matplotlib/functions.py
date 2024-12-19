from typing import List


def sum_non_neg_diag(X: List[List[int]]) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    sum = 0
    fl = False
    for i in range(len(X)):
        for j in range(len(X[i])):
            if i == j and X[i][j] >= 0:
                sum += X[i][j]
                fl = True

    if fl:
        return sum
    return -1


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    return sorted(x) == sorted(y)


def max_prod_mod_3(x: List[int]) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    mul = -1
    for i in range(1, len(x)):
        if x[i] % 3 == 0 or x[i - 1] % 3 == 0:
            mul = max(mul, x[i] * x[i - 1])
    return mul


def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    """
    Сложить каналы изображения с указанными весами.
    """
    l = [0] * len(image)
    for i in range(0, len(image)):
        l[i] = [0] * len(image[0])
        for j in range(0, len(image[0])):
            sum = 0
            for k in range(0, len(image[0][0])):
                sum += image[i][j][k] * weights[k]
            l[i][j] = sum
    return l


def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    x_list = []
    y_list = []
    count = 0
    count_x, count_y = 0, 0
    while count < len(x):
        if count_x < x[count][1]:
            count_x += 1
            x_list.append(x[count][0])
        else:
            count_x = 1
            if count + 1 < len(x):
                x_list.append(x[count + 1][0])
            count += 1

    count = 0
    while count < len(y):
        if count_y < y[count][1]:
            count_y += 1
            y_list.append(y[count][0])
        else:
            count_y = 1
            if count + 1 < len(y):
                y_list.append(y[count + 1][0])
            count += 1

    if len(x_list) != len(y_list):
        return -1

    sum = 0
    # print(x, y, x_list, y_list)
    for i in range(len(x_list)):
        sum += x_list[i] * y_list[i]
    # print(sum)
    return sum


def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y. 
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    l = [0] * len(X)
    for i in range(len(X)):
        l[i] = [0] * len(Y)
        for j in range(len(Y)):
            nums = 0
            x2 = 0
            y2 = 0
            for k in range(len(X[i])):
                nums += X[i][k] * Y[j][k]
                x2 += X[i][k] * X[i][k]
                y2 += Y[j][k] * Y[j][k]
            if x2 == 0 or y2 == 0:
                l[i][j] = 1
            else:
                l[i][j] = nums / (x2 ** 0.5 * y2 ** 0.5)
    return l
