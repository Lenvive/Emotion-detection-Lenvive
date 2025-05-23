import os
import cv2
from utils import ensure_dir, load_images_from_folder, get_emotion_folders
from feature_extractor import FacialFeatureExtractor

def process_dataset(input_base, output_base):
    """
    处理整个数据集
    
    参数:
        input_base: 输入数据集的根目录
        output_base: 输出特征数据的根目录
    """
    extractor = FacialFeatureExtractor(output_size=(64, 64))
    
    # 获取所有情感文件夹
    emotion_folders = get_emotion_folders(input_base)
    
    for emotion, input_folder in emotion_folders.items():
        print(f"Processing {emotion}...")
        
        # 确保输出目录存在
        output_folder = os.path.join(output_base, emotion)
        ensure_dir(output_folder)
        
        # 加载所有图像
        images = load_images_from_folder(input_folder)
        
        # 处理每张图像
        for i, img in enumerate(images):
            features = extractor.extract_features(img)
            
            if features is not None:
                # 保存每个特征
                for feature_name, feature_img in features.items():
                    feature_output_dir = os.path.join(output_folder, feature_name)
                    ensure_dir(feature_output_dir)
                    
                    output_path = os.path.join(feature_output_dir, f"img{i}.png")
                    cv2.imwrite(output_path, feature_img)

if __name__ == "__main__":
    # 处理训练集和测试集
    print("Processing training data...")
    process_dataset(
        input_base="data/train",
        output_base="data_partical/train_features"
    )
    
    print("\nProcessing test data...")
    process_dataset(
        input_base="data/test",
        output_base="data_partical/test_features"
    )
    
    print("\nFeature extraction completed!")
