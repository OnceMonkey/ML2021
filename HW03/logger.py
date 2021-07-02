import torch
import os
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
import mlflow



class Mlflow_Logger():
    def __init__(self, track_path='./mlruns',experiment_name='Default',run_name=None):
        mlflow.set_tracking_uri(track_path)
        mlflow.set_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        runer=mlflow.start_run(experiment_id=experiment.experiment_id,run_name=run_name)

    def __del__(self):
        mlflow.end_run()

    def log_metric(self,key,value,step=None):
        mlflow.log_metric(key,value,step)

    def log_params(self,params):
        mlflow.log_params(params)

    def log_param(self,key,value):
        mlflow.log_param(key,value)
    '''
    image: tensor [C,H,W]
    保存时必须是numpy格式, 因此需要对图片进行一次格式转换
    '''
    def log_image(self,image,artifact_file):
        mlflow.log_image(ToPILImage()(image),artifact_file)
    '''
    保存图片组
    images: tensor [B,C,H,W]
    '''
    def log_images(self,images,artifact_file,nrow=8):
        image_grid=make_grid(images,nrow)
        mlflow.log_image(ToPILImage()(image_grid), artifact_file)
    '''
    figure: plt.figure()
    '''
    def log_figure(self,figure,artifact_file):
        mlflow.log_figure(figure,artifact_file)

    def getdir_artifact(self, mode='relative'):
        dst_dir = mlflow.get_artifact_uri().replace('file:///', '')
        if mode == 'relative':
            return os.path.relpath(dst_dir)
        else:
            return dst_dir