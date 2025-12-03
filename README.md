train
python train_model.py --dataset_path D:\anomaly\normal_images_expanded --model_save_path D:\anomaly\model4tyre_v2 --nu 0.3 --gamma 0.01

test
python test_model_fixed.py --model_path D:\anomaly\model4tyre_v2 --test_folder D:\anomaly\tyretest
