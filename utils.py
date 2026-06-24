from tqdm import tqdm
import numpy as np
from typing import Literal, List, Sequence, Tuple
from sklearn.metrics import f1_score, recall_score, precision_score, matthews_corrcoef, confusion_matrix, roc_auc_score, precision_recall_curve
import torch
import torch.nn as nn
import pandas as pd
import CARZero

def split_list(lst, chunk_size):
    result = []
    for i in range(0, len(lst), chunk_size):
        chunk = lst[i:i+chunk_size]
        result.append(chunk)
    return result

def learn_temperature_multilabel_scalar(sim_logits, y_true, device,
                                        max_iter=100, lr=0.01):
    logits = torch.tensor(sim_logits, dtype=torch.float32, device=device)
    targets = torch.tensor(y_true, dtype=torch.float32, device=device)

    # temperature 파라미터 (scalar)
    temperature = nn.Parameter(torch.ones(1, device=device))

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        T = temperature.clamp(min=1e-6)
        scaled_logits = logits / T
        loss = criterion(scaled_logits, targets)
        loss.backward()
        return loss

    optimizer.step(closure)

    T_star = float(temperature.detach().item())
    sim_scaled = sim_logits / T_star

    return T_star, sim_scaled

def obtain_attn(df, texts, CARZero_model, device, multi=True, mcq=True): 
    # process input images and class prompts 
    ## batchsize
    bs = 256
    image_list = split_list(df['Path'].tolist(), bs)
    processed_txt = CARZero_model.process_text(texts, device)
    
    sim = []
    i2t_attns = []
    t2i_attns = []
    for i, img in tqdm(enumerate(image_list), total=len(image_list), desc="Processing images"):
        processed_imgs = CARZero_model.process_img(img, device)
        # zero-shot classification on 1000 images
        similarities, i2t_attn, t2i_attn  = CARZero.dqn_shot_classification(
            CARZero_model, processed_imgs, processed_txt, atten_map=True, multi=multi, mcq=mcq)
        
        sim.append(similarities)
        i2t_attns.append(i2t_attn.squeeze())
        t2i_attns.append(np.transpose(t2i_attn, (0, 2, 1, 3)).squeeze())

    sim = np.concatenate(sim, axis=0)
    i2t_attns = np.concatenate(i2t_attns, axis=0)
    t2i_attns = np.concatenate(t2i_attns, axis=0)
    
    return sim, i2t_attns, t2i_attns

def calculate_pnc_logit(
    pos_logits: np.ndarray,
    neg_logits: np.ndarray,
    reduction: Literal["none", "mean"] = "none",
) -> np.ndarray:
    if pos_logits.shape != neg_logits.shape:
        raise ValueError(
            f"Shape mismatch: pos_logits {pos_logits.shape}, "
            f"neg_logits {neg_logits.shape}"
        )

    # (..., 2)축으로 결합: 마지막 축 [-1] = positive, [0] = negative
    logits = np.stack([neg_logits, pos_logits], axis=-1)

    # --- Softmax 계산 ---
    # 안정성을 위해 log-sum-exp 사용
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # positive class 확률 = 마지막 축의 index 1
    prob_pos = probs[..., 1]

    if reduction == "mean":
        return prob_pos.mean()
    elif reduction == "none":
        return prob_pos
    else:
        raise ValueError("reduction must be 'none' or 'mean'")

def calculate_metric(logit, true_labels, class_name, nf=False, neg=False) :
    dic = {"accuracy": [], "f1_score": [], "recall": [], "precision": [], "auc": [], "mcc": []}
    sigmoid_output = torch.sigmoid(torch.tensor(logit)).numpy()
    for i, class_label in enumerate(tqdm(class_name)):
        if not nf :
            if i == len(class_name) - 1:  # 'No Finding' 클래스는 제외
                continue
        # 실제값과 예측값 비교
        true_label = true_labels[:,i]
        if neg :
            true_label = 1 - true_label
            
        #mccs,threshold = compute_mccs(true_label.reshape(-1, 1), sigmoid_output[:, i].reshape(-1,1))
        precision, recall, thresholds = precision_recall_curve(true_label, sigmoid_output[:, i])
        numerator = 2 * recall * precision
        denom = recall + precision
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
        max_f1 = np.max(f1_scores)
        max_f1_thresh = thresholds[np.argmax(f1_scores)]
        predicted_labels = (sigmoid_output[:, i] > max_f1_thresh).astype(int)  # 예측값 (threshold 사용)
        
        # 정확도 계산
        accuracy = np.mean(true_label == predicted_labels)
        # F1-score, Recall, Precision, and AUC 계산
        f1 = f1_score(true_label, predicted_labels)
        recall = recall_score(true_label, predicted_labels)
        precision = precision_score(true_label, predicted_labels)
        auc = roc_auc_score(true_label, sigmoid_output[:, i])
        # Confusion matrix 계산
        cm = confusion_matrix(true_label, predicted_labels)
        mcc = matthews_corrcoef(true_label, predicted_labels)
        
        dic["accuracy"].append(accuracy)
        dic["f1_score"].append(f1)
        dic["recall"].append(recall)
        dic["precision"].append(precision)
        dic["auc"].append(auc)
        dic["mcc"].append(mcc)
    
    results = pd.DataFrame(dic, index=class_name[:-1]) if not nf else pd.DataFrame(dic, index=class_name)
    # 마지막 행을 제외한 나머지 행들의 평균을 'Mean' 행으로 추가
    if nf:
        mean_values = results.iloc[:-1].mean(numeric_only=True)
    else:
        mean_values = results.mean(numeric_only=True)
    results.loc['Mean'] = mean_values
            
    return results