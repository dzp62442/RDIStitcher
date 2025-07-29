# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import imageio
from .Warp.Codes.network import build_output_model, WarpNetwork
import os
import cv2
from tqdm import tqdm
import setproctitle
from loguru import logger

from .Warp.Codes import grid_res
grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))  # UDIS2 项目文件夹
DATASET_ROOT = "/home/B_UserData/dongzhipeng/Datasets"
MODEL_DIR = os.path.join(PROJ_ROOT, 'sv_comp/Warp/model/')



class UDIS2Warp:
    def __init__(self, config: dict):
        self.config = config
        self.net = WarpNetwork()
        if torch.cuda.is_available():
            self.net = self.net.cuda()
        self.net.eval()

        #load the existing models if it exists
        model_path = os.path.join(MODEL_DIR, config['model'])
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            self.net.load_state_dict(checkpoint['model'])
            logger.info('load model from {} !'.format(model_path))
        else:
            logger.error('No pretrained model {} !'.format(model_path))
        

    def test(self, input_tensors):
        out_dict = {}  # 存储输出结果
        inpu1_tesnor = input_tensors[0].float()
        inpu2_tesnor = input_tensors[1].float()

        if torch.cuda.is_available():
            inpu1_tesnor = inpu1_tesnor.cuda()
            inpu2_tesnor = inpu2_tesnor.cuda()

        with torch.no_grad():
            batch_out = build_output_model(self.net, inpu1_tesnor, inpu2_tesnor)

        if (not batch_out['success']):  # 拼接失败情况处理
            out_dict.update(success=False)
            return out_dict

        final_warp1 = batch_out['final_warp1']
        final_warp1_mask = batch_out['final_warp1_mask']
        final_warp2 = batch_out['final_warp2']
        final_warp2_mask = batch_out['final_warp2_mask']
        final_mesh1 = batch_out['mesh1']
        final_mesh2 = batch_out['mesh2']

        final_warp1 = ((final_warp1[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        final_warp2 = ((final_warp2[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        final_warp1_mask = final_warp1_mask[0].cpu().detach().numpy().transpose(1,2,0)
        final_warp2_mask = final_warp2_mask[0].cpu().detach().numpy().transpose(1,2,0)
        final_mesh1 = final_mesh1[0].cpu().detach().numpy()
        final_mesh2 = final_mesh2[0].cpu().detach().numpy()

        torch.cuda.empty_cache()

        out_dict.update(success=True, warp1=final_warp1, mask1=final_warp1_mask, warp2=final_warp2, mask2=final_warp2_mask, mesh1=final_mesh1, mesh2=final_mesh2)

        return out_dict