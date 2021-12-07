from augment_utils.utils import augment_image
from augment_utils.split_texture import split_texture
from kmeans_utils.generate_brush import applyBilateralFilterFromImagePath, kMeansClusteringLAB, kMeansClusteringShapeDetection, extractMaskBoundayAndBrushData


if __name__ == '__main__':
    input_dir = './test/test_texture'
    output_dir = './test/test_result'
    height, width = 256,256
    crops = split_texture(input_dir=input_dir, output_dir=output_dir, height=height, width=width)
    augments = augment_image(crops=crops)

    for a in augments:
        blur = applyBilateralFilterFromImagePath(a)
        clusters = kMeansClusteringLAB(blur)
        cluster_data = kMeansClusteringShapeDetection(blur, clusters, 3, 3, 10)
        extractMaskBoundayAndBrushData(input_image=a, blur_image=blur, cluster_data=cluster_data, output_dir=output_dir)
    