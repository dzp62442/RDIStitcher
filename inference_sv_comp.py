import os
import argparse
import torch
import numpy as np
import cv2
import random

from diffusers import UniPCMultistepScheduler
from diffusers import StableDiffusionInpaintPipeline

from procossing import preprocessing_test

from omegaconf import OmegaConf
from sv_comp.udis2_warp import UDIS2Warp
from sv_comp.Warp.Codes.dataset import MultiWarpDataset
import yaml
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(description="RDIStitcher.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="models/stable-diffusion-2-inpainting"
    )
    parser.add_argument(
        "--loramodel_path",
        type=str,
        default="loraweight",
    )
    parser.add_argument(
        "--test_prompt",
        type=str,
        default="<A>",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="examples"
    )
    parser.add_argument(
        "--udis_cfg",
        type=str,
        default="sv_comp/udis2_warp.yaml"
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="results",
    )
    parser.add_argument(
        "--num_seed",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    return args


def main(args):
    # 初始化 UDIS2 的 warp 网络
    udis2_warp_cfg = OmegaConf.load(args.udis_cfg)
    udis2_warp = UDIS2Warp(udis2_warp_cfg)

    # 加载数据集
    with open('sv_comp/intrinsics.yaml', 'r', encoding='utf-8') as file:
        intrinsics = yaml.safe_load(file)
    dataset = MultiWarpDataset(config=udis2_warp_cfg, intrinsics=intrinsics, is_train=False)

    # 初始化 SRStitcher 网络
    device = args.device
    pretrained_model_name_or_path = args.pretrained_model_name_or_path
    loramodel_path = args.loramodel_path
    data_root = args.data_root
    save_root = args.save_root
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        pretrained_model_name_or_path, torch_dtype=torch.float16
    )
    pipe.load_lora_weights(loramodel_path)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    
    generator = torch.manual_seed(3407)  # 固定随机种子
    
    failure_num = 0  # 拼接失败的次数
    for idx in range(len(dataset)):
        if idx > 2:
            break
        sample = dataset[idx]
        input_imgs, input_masks = sample[0], sample[1]
        middle_stitch_result = None  # 中间拼接结果

        for k in range(udis2_warp_cfg['input_img_num'] - 1):
            # 创建保存结果文件夹
            batch_path = dataset.get_path(idx)
            path_result = os.path.join(batch_path, 'rdistitcher/')
            os.makedirs(path_result, exist_ok=True)
            path_warp = os.path.join(batch_path, 'warp/')
            os.makedirs(path_warp, exist_ok=True)
            path_mask = os.path.join(batch_path, 'mask/')
            os.makedirs(path_mask, exist_ok=True)
            
            # 使用 UDIS2 的 warp 网络进行拼接
            if k == 0:
                input1 = dataset.to_tensor(input_imgs[k])
                input2 = dataset.to_tensor(input_imgs[k+1])
            else:
                input1 = dataset.to_tensor(middle_stitch_result)
                input2 = dataset.to_tensor(input_imgs[k+1])
            out_dict = udis2_warp.test([input1, input2])
            if (not out_dict['success']):  # 拼接失败情况处理
                logger.warning(f'Failed to stitch {dataset.get_path(idx)} !!!')
                logger.warning(f'batch {idx} stitch fail when k={k} !!!')
                failure_num += 1
                torch.cuda.empty_cache()
                break
            warp1 = out_dict['warp1'].astype(np.uint8)
            warp2 = out_dict['warp2'].astype(np.uint8)
            mask1 = cv2.cvtColor(out_dict['mask1'].astype(np.uint8), cv2.COLOR_BGR2GRAY)*255
            mask2 = cv2.cvtColor(out_dict['mask2'].astype(np.uint8), cv2.COLOR_BGR2GRAY)*255
            
            # 掩码中白色区域可能存在细小黑点，通过形态学操作去除
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 形态学操作去除细小黑点
            mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)  # 闭操作：先膨胀后腐蚀，填充小的黑点（在白色区域中）
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
            mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)  # 开操作：先腐蚀后膨胀，去除小的白点（在黑色背景上）
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
            
            # 保存 warp 和 mask
            name = f"{udis2_warp_cfg['input_img_num']}_{k+2}.jpg"
            cv2.imwrite(os.path.join(path_warp, name.replace('.jpg', '_warp1.jpg')), warp1)
            cv2.imwrite(os.path.join(path_mask, name.replace('.jpg', '_mask1.jpg')), mask1)
            cv2.imwrite(os.path.join(path_warp, name.replace('.jpg', '_warp2.jpg')), warp2)
            cv2.imwrite(os.path.join(path_mask, name.replace('.jpg', '_mask2.jpg')), mask2)

            h,w = warp1.shape[0], warp1.shape[1]

            image, mask_image = preprocessing_test(warp1,warp2,mask1,mask2)

            text_prompt=args.test_prompt

            new_image = pipe(
                        prompt=text_prompt,
                        num_inference_steps=50,
                        generator=generator,
                        image=image,
                        mask_image=mask_image,
                        height = 512,
                        width =1024,
                    ).images[0]
            new_image = new_image.crop((512, 0, 1024, 512))
            
            # 处理中间拼接结果
            middle_stitch_result = np.array(new_image)
            middle_stitch_result = cv2.resize(middle_stitch_result, (udis2_warp_cfg['net_input_width'], udis2_warp_cfg['net_input_height']))
            middle_stitch_result = cv2.cvtColor(middle_stitch_result, cv2.COLOR_RGB2BGR)

            # 保存结果
            new_image = new_image.resize((w,h))
            name = f"{udis2_warp_cfg['input_img_num']}_{k+2}.jpg"
            new_image.save(os.path.join(path_result, name))

        print(f'processing image {idx}/{len(dataset)} completed')


if __name__ == "__main__":
    args = parse_args()
    main(args)
