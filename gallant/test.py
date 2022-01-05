from augment_image import augment_image
from split_texture import split_texture
from generate_brush import applyBilateralFilterFromImage, kMeansClusteringLAB, kMeansClusteringShapeDetection, extractMaskBoundaryAndBrushData


if __name__ == '__main__':
    input_dir = './test_texture'
    output_dir = './test_result'
    height, width = 256,256
    crops = split_texture(input_dir=input_dir, output_dir=output_dir, height=height, width=width)
    # augments = augment_image(crops=crops)

    for a in crops:
        blur = applyBilateralFilterFromImage(a)
        clusters, n = kMeansClusteringLAB(blur)
        cluster_data = kMeansClusteringShapeDetection(blur, clusters, 3, 3, 10, n)
        extractMaskBoundaryAndBrushData(input_image=a, blur_image=blur, cluster_data=cluster_data, output_dir=output_dir)