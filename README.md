Autoencoder
train:
python trainauto2.py --dataset_path D:\anomaly\tyregood --model_save_path D:\anomaly\model
auto2 --threshold_percentile 98 

test:
python testauto2.py --model_path D:\anomaly\modelauto2 --test_folder D:\anomaly\tyretest -
-custom_threshold -5.0 






train :   

python train_model.py --dataset_path D:\anomaly\normal_images_expanded --model_save_path D:\anomaly\model4tyre_v2 --nu 0.3 --gamma 0.01

test:

python test_model_fixed.py --model_path D:\anomaly\model4tyre_v2 --test_folder D:\anomaly\tyretest




