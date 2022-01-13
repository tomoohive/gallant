import os
import cv2
import numpy as np

from sklearn.cluster import KMeans

# from utils import rand_color_map
from random_name import random_name

def applyBilateralFilterFromImagePath(image_path: str):
    blur = cv2.bilateralFilter(cv2.imread(image_path), 9, 75, 75)
    return blur


def applyBilateralFilterFromImage(image: np.ndarray):
    blur = cv2.bilateralFilter(image, 9, 75, 75)
    return blur


def kMeansClusteringLAB(input_image: np.ndarray, l_weight=1, a_weight=5, b_weight=5):
    input_shape = input_image.shape
    lab_data = np.empty((0,3), int)
    input_image_lab = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB)
    n = 0
    # input_image_hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    l, a, b = cv2.split(input_image_lab)
    
    for index_x in range(input_image_lab.shape[0]):
        lab_datum = np.empty((0,3), int)
        for index_y in range(input_image_lab.shape[1]):
            lab = np.array([[l[index_x][index_y]*l_weight, a[index_x][index_y]*a_weight, b[index_x][index_y]*b_weight]])
            lab_datum = np.append(lab_datum, lab, axis=0)
        lab_data = np.concatenate((lab_data, lab_datum), axis=0)

    kmeans = KMeans(n_clusters=2, init='k-means++', n_jobs=2).fit(lab_data)
    kmeans.fit(lab_data)
    kmeans_clusters = kmeans.predict(lab_data)

    clusters = np.split(kmeans_clusters, int(len(kmeans_clusters)/input_shape[1]))
    a_1 = kmeans.cluster_centers_[0][1]
    a_2 = kmeans.cluster_centers_[1][1]
    if(637.5-a_1 > 637.5-a_2):
        n = 0
    else:
        n = 1
    return clusters, n


def kMeansClusteringShapeDetection(input_image, splits_clusters, h_weight=1, s_weight=1, n=50, channel=0):
    input_shape = input_image.shape
    input_image_hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(input_image_hsv)
    nh = (h / 180).astype(float)
    ns = (h / 255).astype(float)
    nv = (v / 255).astype(float)
    texture_data = np.empty((0,4), int)
    coordinate_data = np.empty((0,2), int)

    for x, split_clusters in enumerate(splits_clusters):
        texture_datum = np.empty((0,4), int)
        coordinate_datum = np.empty((0,2), int)
        for y, split_cluster in enumerate(split_clusters):
            if split_cluster == channel:
                texture_dot = np.array([[x, y, h[x][y]*h_weight, s[x][y]*s_weight]])
                texture_datum = np.append(texture_datum, texture_dot, axis=0)
                coordinate = np.array([[x, y]])
                coordinate_datum = np.append(coordinate_datum, coordinate, axis=0)
        texture_data = np.concatenate((texture_data, texture_datum), axis=0)
        coordinate_data = np.concatenate((coordinate_data, coordinate_datum), axis=0)

    kmeans = KMeans(n_clusters= n)
    kmeans.fit(texture_data)
    kmeans_clusters = kmeans.predict(texture_data)

    cluster_data = {}
    for coordinate, cluster in zip(coordinate_data, kmeans_clusters):
        if str(cluster) not in cluster_data:
            cluster_data[str(cluster)] = []
        cluster_data[str(cluster)].append([int(coordinate[0]), int(coordinate[1])])

    # RGB = rand_color_map(0, 255, n)
    
    # cluster_layer = np.zeros(shape=(input_shape[0], input_shape[1], 4), dtype=int)
    # for coordinate, color_map in zip(coordinate_data, kmeans_clusters):
    #     rgb = RGB[color_map - 1]
    #     cluster_layer[coordinate[0]][coordinate[1]] = np.array([rgb[0], rgb[1], rgb[2], 255])
    # cv2.imwrite(output_dir + '/mask.png', cluster_layer[:,:,:3])
    # cv2.imwrite(output_dir + '/layer.png', cluster_layer)

    return cluster_data


# def extractMaskBoundayAndBrushData(input_path, cluster_data):
#     MASK_CROP_DATA_DIRECTORY = output_dir + '/mask'
#     BRUSH_CROP_DATA_DIRECTORY = output_dir + '/crop'

#     r, g, b = cv2.split(cv2.imread(input_path))
#     image_shape = r.shape
#     for index, cluster_index in enumerate(cluster_data):
#         cluster_datum = np.array(cluster_data[cluster_index])
#         xy_max = np.max(cluster_datum, axis=0)
#         xy_min = np.min(cluster_datum, axis=0)
#         mask_layer = np.zeros(shape=(image_shape[0], image_shape[1], 3), dtype=int)
#         brush_layer = np.zeros(shape=(image_shape[0], image_shape[1], 4), dtype=int)
#         for coordinate in cluster_data[cluster_index]:
#             x, y =  coordinate[0], coordinate[1]
#             mask_layer[x][y] = np.array([255, 255, 255])
#             brush_layer[x][y] = np.array([r[x][y], g[x][y], b[x][y], 255])
#         try:
#             cv2.imwrite(MASK_CROP_DATA_DIRECTORY + '/mask_crop' + str(index) + '.png', mask_layer)
#             cv2.imwrite(BRUSH_CROP_DATA_DIRECTORY + '/brush_crop' + str(index) + '.png', brush_layer[int(xy_min[0]):int(xy_max[0]), int(xy_min[1]):int(xy_max[1])])
#         except:
#             pass

    
def extractMaskBoundaryAndBrushData(input_image: np.ndarray, blur_image: np.ndarray, cluster_data: np.array, output_dir):
    name = random_name(10)
    IMAGE_DATA_DIRECTORY = output_dir + '/' + name
    MASK_CROP_DATA_DIRECTORY = IMAGE_DATA_DIRECTORY + '/mask'
    BRUSH_CROP_DATA_DIRECTORY = IMAGE_DATA_DIRECTORY + '/crop'

    if not os.path.exists(IMAGE_DATA_DIRECTORY):
        os.mkdir(IMAGE_DATA_DIRECTORY)

    if not os.path.exists(MASK_CROP_DATA_DIRECTORY):
        os.mkdir(MASK_CROP_DATA_DIRECTORY)

    if not os.path.exists(BRUSH_CROP_DATA_DIRECTORY):
        os.mkdir(BRUSH_CROP_DATA_DIRECTORY)

    r, g, b = cv2.split(input_image)
    image_shape = r.shape
    for index, cluster_index in enumerate(cluster_data):
        cluster_datum = np.array(cluster_data[cluster_index])
        xy_max = np.max(cluster_datum, axis=0)
        xy_min = np.min(cluster_datum, axis=0)
        mask_layer = np.zeros(shape=(image_shape[0], image_shape[1], 3), dtype=int)
        brush_layer = np.zeros(shape=(image_shape[0], image_shape[1], 4), dtype=int)
        for coordinate in cluster_data[cluster_index]:
            x, y =  coordinate[0], coordinate[1]
            mask_layer[x][y] = np.array([255, 255, 255])
            brush_layer[x][y] = np.array([r[x][y], g[x][y], b[x][y], 255])
        try:
            cv2.imwrite(MASK_CROP_DATA_DIRECTORY + '/mask_crop' + str(index) + '.png', mask_layer)
            cv2.imwrite(BRUSH_CROP_DATA_DIRECTORY + '/brush_crop' + str(index) + '.png', brush_layer[int(xy_min[0]):int(xy_max[0]), int(xy_min[1]):int(xy_max[1])])
        except:
            pass
    cv2.imwrite(IMAGE_DATA_DIRECTORY + '/' + name + '.png', input_image)
    cv2.imwrite(IMAGE_DATA_DIRECTORY + '/blur_' + name + '.png', blur_image)


# if __name__ == '__main__':
#     image_path = './input.png'
#     output_dir = './test'

#     blur = applyBilateralFilterFromImagePath(image_path)
#     clusters = kMeansClusteringLAB(blur)
#     cluster_data = kMeansClusteringShapeDetection(blur, clusters, 3, 3)
#     extractMaskBoundayAndBrushData(image_path, cluster_data)