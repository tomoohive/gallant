from augment_utils.utils import augment_image
from augment_utils.split_texture import split_texture
from kmeans_utils.generate_brush import applyBilateralFilterFromImagePath, kMeansClusteringLAB, kMeansClusteringShapeDetection, extractMaskBoundayAndBrushData


if __name__ == '__main__':
    image_path = './input.png'
    
    output_dir = './test/test_result'

    blur = applyBilateralFilterFromImagePath(image_path)
    clusters = kMeansClusteringLAB(blur)
    cluster_data = kMeansClusteringShapeDetection(blur, clusters, 3, 3)
    extractMaskBoundayAndBrushData(image_path, cluster_data)

    input_dir = './moss_textures'
    output_dir = './result'
    height, width = 512, 512

    split_texture(input_dir=input_dir, output_dir=output_dir, height=height, width=width)