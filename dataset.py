from einops import rearrange
import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.dataloader import DataLoader
# from .datasets import Kinetics
# from .datasets import UCF101
from datasets import Jester


def get_training_set(opt, spatial_transform, temporal_transform,
					 target_transform):
	assert opt.dataset in ['kinetics', 'jester', 'ucf101']

	if opt.dataset == 'kinetics':
		training_data = Kinetics(
			opt.video_path,
			opt.annotation_path,
			'training',
			spatial_transform=spatial_transform,
			temporal_transform=temporal_transform,
			target_transform=target_transform,
			sample_duration=opt.sample_duration)
	elif opt.dataset == 'jester':
		training_data = Jester(
			opt.video_path,
			opt.annotation_path,
			'training',
			n_samples_for_each_video=1,
			spatial_transform=spatial_transform,
			temporal_transform=temporal_transform,
			target_transform=target_transform,
			sample_duration=opt.sample_duration)
	elif opt.dataset == 'ucf101':
		training_data = UCF101(
			opt.video_path,
			opt.annotation_path,
			'training',
			spatial_transform=spatial_transform,
			temporal_transform=temporal_transform,
			target_transform=target_transform,
			sample_duration=opt.sample_duration)
	return training_data


def get_validation_set(opt, spatial_transform, temporal_transform,
					   target_transform):
	assert opt.dataset in ['kinetics', 'jester', 'ucf101']

	if opt.dataset == 'kinetics':
		validation_data = Kinetics(
			opt.video_path,
			opt.annotation_path,
			'validation',
			opt.n_val_samples,
			spatial_transform,
			temporal_transform,
			target_transform,
			sample_duration=opt.sample_duration)
	elif opt.dataset == 'jester':
		validation_data = Jester(
			opt.video_path,
			opt.annotation_path,
			'validation',
			opt.n_val_samples,
			spatial_transform,
			temporal_transform,
			target_transform,
			sample_duration=opt.sample_duration)
	elif opt.dataset == 'ucf101':
		validation_data = UCF101(
			opt.video_path,
			opt.annotation_path,
			'validation',
			opt.n_val_samples,
			spatial_transform,
			temporal_transform,
			target_transform,
			sample_duration=opt.sample_duration)
	return validation_data


def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
	assert opt.dataset in ['kinetics', 'jester', 'ucf101']
	assert opt.test_subset in ['val', 'test']

	if opt.test_subset == 'val':
		subset = 'validation'
	elif opt.test_subset == 'test':
		subset = 'testing'
	if opt.dataset == 'kinetics':
		test_data = Kinetics(
			opt.video_path,
			opt.annotation_path,
			subset,
			0,
			spatial_transform,
			temporal_transform,
			target_transform,
			sample_duration=opt.sample_duration)
	elif opt.dataset == 'jester':
		test_data = Jester(
			opt.video_path,
			opt.annotation_path,
			subset,
			0,
			spatial_transform,
			temporal_transform,
			target_transform,
			sample_duration=opt.sample_duration)
	elif opt.dataset == 'ucf101':
		test_data = UCF101(
			opt.video_path,
			opt.annotation_path,
			subset,
			0,
			spatial_transform,
			temporal_transform,
			target_transform,
			sample_duration=opt.sample_duration)
	return test_data

class Collator(object):

	def __init__(self):
		pass
  	
	def collate(self, minibatch):
		image_list = []
		label_list = []
		for record in minibatch:
			# Filter out records that aren't load correctly.
			if record[-1] == -1:
				del record
				continue
			else:
				image_list.append(record[0])
				label_list.append(record[1])
		minibatch = []
		minibatch.append(torch.stack(image_list))
		label = np.stack(label_list)
		minibatch.append(torch.from_numpy(label))
		return minibatch


class ActionDataModule(pl.LightningDataModule):
	def __init__(self, 
				 opt,
				 train_spatial_transform, 
				 train_temporal_transform,
				 train_target_transform,
				 val_spatial_transform = None, 
				 val_temporal_transform = None,
				 val_target_transform = None,
				 test_spatial_transform = None, 
				 test_temporal_transform = None,
				 test_target_transform = None):
		super().__init__()
		
		self.opt = opt
		self.train_spatial_transform = train_spatial_transform
		self.train_temporal_transform = train_temporal_transform
		self.train_target_transform = train_target_transform
		self.val_spatial_transform = val_spatial_transform
		self.val_temporal_transform = val_temporal_transform 	
		self.val_target_transform = val_target_transform
		self.test_spatial_transform = test_spatial_transform
		self.test_temporal_transform = test_temporal_transform
		self.test_target_transform = test_target_transform


	def setup(self, stage):
		if not self.opt.no_train:
			self.train_dataset = get_training_set(
				self.opt,
				self.train_spatial_transform,
				self.train_temporal_transform,
				self.train_target_transform)
		
		if not self.opt.no_val:
			self.val_dataset = get_validation_set(
				self.opt,
				self.val_spatial_transform,
				self.val_temporal_transform,
				self.val_target_transform)
		
		if self.opt.test:
			self.val_dataset = get_validation_set(
				self.opt,
				self.test_spatial_transform,
				self.test_temporal_transform,
				self.test_target_transform)


	def train_dataloader(self):
		return DataLoader(
			self.train_dataset,
			batch_size=self.opt.batch_size,
			num_workers=self.opt.num_workers,
# 			collate_fn=Collator().collate,
			shuffle=True,
# 			drop_last=True, 
			pin_memory=True
		)
	
	def val_dataloader(self):
		if self.val_spatial_transform is not None:
			return DataLoader(
				self.val_dataset,
				batch_size=self.opt.batch_size // 2,
				num_workers=self.opt.num_workers,
# 				collate_fn=Collator().collate,
				shuffle=False,
# 				drop_last=False,
			)
	
	def test_dataloader(self):
		if self.test_spatial_transform is not None:
			return DataLoader(
				self.test_dataset,
				batch_size=self.opt.batch_size // 2,
				num_workers=self.opt.num_workers,
# 				collate_fn=Collator().collate,
				shuffle=False,
# 				drop_last=False,
			)
