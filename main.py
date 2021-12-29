import comet_ml
import os
import time
import random
import warnings
import json
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
import torch.utils.data as data
from dataset import ActionDataModule
from utils import print_on_rank_zero, get_mean, get_std
from opts import parse_opts
from spatial_transforms import *
from temporal_transforms import *
from target_transforms import Compose as TargetCompose
from target_transforms import ClassLabel, VideoID
from models import VideoTransformer

def run():
	args = parse_opts()
	warnings.filterwarnings('ignore')
	
	ROOT_DIR = args.result_path
	os.makedirs(ROOT_DIR, exist_ok=True)
		
	exp_tag = f'arch_{args.model}_nframe_{args.num_frames}_seed_{args.manual_seed}'
	ckpt_dir = os.path.join(ROOT_DIR, 'ckpt')
	log_dir = os.path.join(ROOT_DIR, 'log')
	os.makedirs(ckpt_dir, exist_ok=True)
	os.makedirs(log_dir, exist_ok=True)
	args.scales = [args.initial_scale]
	for i in range(1, args.n_scales):
		args.scales.append(args.scales[-1] * args.scale_step)
	args.mean = get_mean(args.norm_value, args.mean_dataset)
	args.std  = get_std(args.norm_value)

	print(args)
	with open(os.path.join(ROOT_DIR, 'opts.json'), 'w') as opt_file:
		json.dump(vars(args), opt_file)
	# train

	if args.no_mean_norm and not args.std_norm:
		norm_method = Normalize([0, 0, 0], [1, 1, 1])
	elif not args.std_norm:
		norm_method = Normalize(args.mean, [1, 1, 1])
	else:
		norm_method = Normalize(args.mean, args.std)
	if not args.no_train:
		assert args.train_crop in ['random', 'corner', 'center']
		if args.train_crop == 'random':
			crop_method = MultiScaleRandomCrop(args.scales, args.img_size)
		elif args.train_crop == 'corner':
			crop_method = MultiScaleCornerCrop(args.scales, args.img_size)
		elif args.train_crop == 'center':
			crop_method = MultiScaleCornerCrop(
				args.scales, args.img_size, crop_positions=['c'])

		train_spatial_transform = Compose([
			RandomHorizontalFlip(),
			#RandomRotate(),
			#RandomResize(),
# 			crop_method,
			#MultiplyValues(),
			#Dropout(),
			#SaltImage(),
			#Gaussian_blur(),
			#SpatialElasticDisplacement(),
			FixResize(args.img_size),
			ToTensor(args.norm_value), norm_method
		])
		train_temporal_transform = TemporalRandomCrop(args.num_frames, args.downsample)
		train_target_transform = ClassLabel()
	else:
		train_spatial_transform = None
		train_temporal_transform = None
		train_target_transform = None


	if not args.no_val:
		do_eval = True
		val_spatial_transform = Compose([
			Scale(args.img_size),
			# CenterCrop(args.img_size),
			FixResize(args.img_size),
			ToTensor(args.img_size), norm_method
		])
		#temporal_transform = LoopPadding(opt.sample_duration)
		val_temporal_transform = TemporalCenterCrop(args.sample_duration, args.downsample)
		val_target_transform = ClassLabel()
	else:
		do_eval = False
		val_spatial_transform = None
		val_temporal_transform = None
		val_target_transform = None

	if args.test:
		do_test = True
		test_spatial_transform = Compose([
			Scale(int(args.img_size / args.scale_in_test)),
# 			CornerCrop(args.img_size, args.crop_position_in_test),
			FixResize(args.img_size),
			ToTensor(args.norm_value), norm_method
		])
		test_temporal_transform = TemporalRandomCrop(args.sample_duration, args.downsample)
		test_target_transform = VideoID()

	else:
		do_test = False
		test_spatial_transform = None
		test_temporal_transform = None
		test_target_transform = None
	
	data_module = ActionDataModule(
				 args,
				 train_spatial_transform = train_spatial_transform, 
				 train_temporal_transform = train_temporal_transform,
				 train_target_transform = train_target_transform,
				 val_spatial_transform = val_spatial_transform, 
				 val_temporal_transform = val_temporal_transform,
				 val_target_transform = val_target_transform,
				 test_spatial_transform = test_spatial_transform, 
				 test_temporal_transform = test_temporal_transform,
				 test_target_transform = test_target_transform)
	
	comet_logger = CometLogger(
		save_dir=log_dir,
		project_name="Transformer in Action Recognition",
		experiment_name=exp_tag,
		offline=True)

	trainer = pl.Trainer(
		gpus=[0], # devices=-1
		accelerator="ddp", # accelerator="gpu",strategy='ddp'
		plugins=DDPPlugin(find_unused_parameters=False),
		max_epochs=args.n_epochs,
		callbacks=[
			LearningRateMonitor("epoch"),
			ModelCheckpoint(every_n_epochs = 1)
		],
		logger=comet_logger,
		check_val_every_n_epoch=1,
		log_every_n_steps=args.log_interval,
		progress_bar_refresh_rate=args.log_interval,
		flush_logs_every_n_steps=args.log_interval*5)
		
	# To be reproducable
	torch.random.manual_seed(args.manual_seed)
	np.random.seed(args.manual_seed)
	random.seed(args.manual_seed)
	pl.seed_everything(args.manual_seed, workers=True)
		
	model = VideoTransformer(trainer, ckpt_dir, do_eval, do_test, args)
	
	timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
	print_on_rank_zero(f'{timestamp} - INFO - Start running,')
	trainer.fit(model, data_module)
	trainer.test(model, data_module)
	
if __name__ == '__main__':
	run()
