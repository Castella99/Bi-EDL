# %%
import os
os.chdir("/shared/home/mai/Taehun/Uncertainty/MICCAI_2025/CARZero")

# %%
import torch
import CARZero
import pandas as pd 
import json
import numpy as np
from utils import *
from sklearn.preprocessing import MultiLabelBinarizer
from glob import glob
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# %%
texts = {"0": ["There is Atelectasis"], "1": ["There is Cardiomegaly"], "2": ["There is Pleural Effusion"], "3": ["There is Pulmonary Infiltration"], "4": ["There is Pulmonary Mass"], "5": ["There is Lung Nodule"], "6": ["There is Pneumonia"], "7": ["There is Pneumothorax"], "8": ["There is Pulmonary Consolidation"], "9": ["There is Pulmonary Edema"], "10": ["There is Pulmonary Emphysema"], "11": ["There is Fibrosis"], "12": ["There is Pleural Thickening"], "13": ["There is Hernia"], "14": ["There is No Finding"]}

def worker(rank, world_size, df_split, texts, return_queue):
    print(f"[RANK {rank}] 시작", flush=True)
    dist.init_process_group(
    backend='nccl',
    init_method='tcp://127.0.0.1:23456',  # 모든 프로세스가 동일한 포트를 공유해야 함
    world_size=world_size,
    rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device=device)
    CARZero_model.eval()
    model = DDP(CARZero_model, device_ids=[rank])
    print(f"[RANK {rank}] 모델 로딩 완료", flush=True)
    
    print(f"[RANK {rank}] process_class_prompts 시작", flush=True)
    processed_txt = model.module.process_class_prompts(texts, device)
    bs = 2048
    print(f"[RANK {rank}] img Load 시작", flush=True)
    image_list = split_list(df_split['Path'].tolist(), bs)
    print(f"[RANK {rank}] img Load 완료", flush=True)
    print(f"[RANK {rank}] sim 계산 시작", flush=True)
    sim_all = []
    for img_paths in image_list:
        imgs = model.module.process_img(img_paths, device)
        print(f"[RANK {rank}] Img Preprocess 완료", flush=True)
        sim = CARZero.dqn_shot_classification(model.module, imgs, processed_txt)
        sim_all.append(sim)
    print(f"[RANK {rank}] sim 계산 완료", flush=True)
    result = pd.concat(sim_all, axis=0)
    return_queue.put(result)
    print(f"[RANK {rank}] 결과 반환 완료", flush=True)
    dist.destroy_process_group()

def obtain_simr_parallel(df, texts, num_gpus=4):
    mp.set_start_method("spawn", force=True)
    world_size = num_gpus
    df_splits = np.array_split(df, world_size)
    return_queue = mp.Queue()

    processes = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, world_size, df_splits[rank], texts, return_queue))
        p.start()
        processes.append(p)

    results = [return_queue.get() for _ in range(world_size)]
    for p in processes:
        p.join()

    final_result = pd.concat(results, axis=0)
    return final_result

# %%
def obtain_simr(df, texts, CARZero_model, device):
    # process input images and class prompts 
    ## batchsize
    bs = 8192
    image_list = split_list(df['Path'].tolist(), bs)
    processed_txt = CARZero_model.process_class_prompts(texts, device)
    for i, img in enumerate(image_list):
        processed_imgs = CARZero_model.process_img(img, device)
        # zero-shot classification on 1000 images
        similarities = CARZero.dqn_shot_classification(
            CARZero_model, processed_imgs, processed_txt)
        
        if i == 0:
            similar = similarities
        else:
            similar = pd.concat([similar, similarities], axis=0)

    return similar

if __name__ == "__main__":
    data_path = '/shared/home/mai/Taehun/Uncertainty/data/NIH'
    with open(os.path.join(data_path, 'test_list.txt'), 'r') as f :
        test_list = f.readlines()
    test_list = [x.strip() for x in test_list]
    path = os.path.join(data_path, 'Data_Entry_2017.csv')
    df = pd.read_csv(path)
    df = df[['Image Index', 'Finding Labels']]
    df = df[df['Image Index'].isin(test_list)]
    img_path = {os.path.basename(x): x for x in glob(os.path.join(data_path, 'images*', '*', '*.png'))}
    df['Path'] = df['Image Index'].map(img_path)
    df['Atelectasis'] = df['Finding Labels'].apply(lambda x: 1 if 'Atelectasis' in x else 0)
    for key, value in texts.items():
        label = value[0].replace("There is ", "").replace(" ", "_")
        df[label] = df['Finding Labels'].apply(lambda x: 1 if label.replace("_", " ") in x else 0)

    workers = torch.cuda.device_count()
    print(f"Number of GPUs: {workers}")

    if workers > 1:
        print("Using multiple GPUs for inference.")
        simr = obtain_simr_parallel(df, texts, num_gpus=workers)
    else:
        print("Using single GPU for inference.")
        CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device="cuda")
        CARZero_model.eval()
        simr = obtain_simr(df, texts, CARZero_model, device="cuda")