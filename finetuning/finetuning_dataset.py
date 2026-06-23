import os
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from glob import glob
from PIL import Image
from pytorch_lightning.core import LightningModule
from torchmetrics.classification import MultilabelAUROC
import CARZero.builder as builder
import CARZero
import pytorch_lightning as pl
import torch.nn.functional as F
import pandas as pd
import numpy as np
import cv2
from nltk.tokenize import RegexpTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from peft.tuners.lora import Linear as LoRALinear
import re
from typing import List, Tuple, Dict, Set, Optional, Sequence, Callable
from torch.utils.data import default_collate

import random, secrets
_R = secrets.SystemRandom()          # cryptographically‑strong independent RNG
from typing import List, Tuple, Any, Dict, Optional

def _pos_sent(disease: str) -> str:
    return f"There is {disease.replace('_', ' ')}."

def _neg_sent(disease: str) -> str:
    return f"There is no {disease.replace('_', ' ')}."

def _hyb_sent(pos: str, neg: str) -> str:
    return f"There is {pos.replace('_', ' ')} but no {neg.replace('_', ' ')}."

def generate_mcq(labels: List[int], class_names: List[str], shuffle=True, no_hyb=False) -> Tuple[List[str], int]:    
    """
    • 문장 3개(POS, NEG, HYB) 생성 후 알고리즘에 따라 정답 타입 지정
    • 반환: choices(List[str] 길이 3), answer_idx(int)
    """
    NF = len(class_names) - 1
    pos_idx = [i for i, y in enumerate(labels) if y == 1 and i != NF]
    neg_idx = [i for i, y in enumerate(labels) if y == 0 and i != NF]

    if no_hyb:
        # ---------- No Finding ----------
        if labels[NF] == 1 or not pos_idx:
            # NEG만이 의미적으로 참
            neg_class = _R.choice(range(NF))
            pos_sent = _pos_sent(class_names[neg_class])  # 오답
            neg_sent = _neg_sent(class_names[neg_class])  # 정답

            choices = [pos_sent, neg_sent]
            if shuffle:
                _R.shuffle(choices)
            return choices, choices.index(neg_sent)

         # ---------- No Finding 아님 ----------
        ans_type = _R.choice(["POS", "NEG"])

        if ans_type == "POS":
            # 정답: 존재하는 클래스의 긍정문
            answer = _pos_sent(class_names[_R.choice(pos_idx)])
            # 오답: 존재하는 클래스의 부정문
            wrong = _neg_sent(class_names[_R.choice(pos_idx)])
        else:
            # 정답: 없는 클래스의 부정문
            answer = _neg_sent(class_names[_R.choice(neg_idx)])
            # 오답: 없는 클래스의 긍정문
            wrong = _pos_sent(class_names[_R.choice(neg_idx)])

        choices = [answer, wrong]
        if shuffle:
            _R.shuffle(choices)
        return choices, choices.index(answer)

    # ─────────────────────────── 1. No-Finding ──────────────────────────
    if labels[NF] == 1 or not pos_idx:
        # 랜덤 클래스 하나 (No Finding 제외)
        # POS / HYB 문장은 아무거나 사용해도 모두 '틀림'
        pos_sent = _pos_sent(class_names[_R.choice(range(NF))])
        neg_sent = _neg_sent(class_names[_R.choice(range(NF))])
        hyb_sent = _hyb_sent(class_names[_R.choice(range(NF))], class_names[_R.choice(range(NF))])
        choices  = [pos_sent, neg_sent, hyb_sent]
        if shuffle:
            _R.shuffle(choices)
        return choices, choices.index(neg_sent)   # 정답 = 부정문

    # ───────────────────── 2. No-Finding 아님 ──────────────────────
    # 2-1. 정답 타입 무작위
    ans_type = _R.choice(["POS", "NEG", "HYB"])

    # 2-2. 각 타입별 문장 정의
    if ans_type == "POS":
        pos_sent = _pos_sent(class_names[_R.choice(pos_idx)])               # ✔ 정답
        neg_sent = _neg_sent(class_names[_R.choice(pos_idx)])               # 오답
        hyb_sent = _hyb_sent(class_names[_R.choice(neg_idx)], class_names[_R.choice(pos_idx)])            # 오답 (n absent, p present → 실제와 반대)
        answer  = pos_sent

    elif ans_type == "NEG":
        pos_sent = _pos_sent(class_names[_R.choice(neg_idx)])               # 오답
        neg_sent = _neg_sent(class_names[_R.choice(neg_idx)])               # ✔ 정답
        hyb_sent = _hyb_sent(class_names[_R.choice(neg_idx)], class_names[_R.choice(pos_idx)])            # 오답
        answer  = neg_sent

    else:  # HYB
        pos_sent = _pos_sent(class_names[_R.choice(neg_idx)])               # 오답
        neg_sent = _neg_sent(class_names[_R.choice(pos_idx)])               # 오답
        hyb_sent = _hyb_sent(class_names[_R.choice(pos_idx)], class_names[_R.choice(neg_idx)])            # ✔ 정답
        answer  = hyb_sent

    choices = [pos_sent, neg_sent, hyb_sent]
    if shuffle:
        _R.shuffle(choices)
    return choices, choices.index(answer)

def generate_mcq2(labels: List[int], class_names: List[str], shuffle=True, no_hyb=False) -> Tuple[List[str], int]:
    """
    • 문장 3개(POS, NEG, HYB) 생성 후 알고리즘에 따라 정답 타입 지정
    • 반환: choices(List[str] 길이 3), answer_idx(int)
    """
    NF = len(class_names) - 1
    pos_idx = [i for i, y in enumerate(labels) if y == 1 and i != NF]
    neg_idx = [i for i, y in enumerate(labels) if y == 0 and i != NF]

    def _rand_disease():
        return class_names[_R.choice(range(NF))]

    def _neg_or_nofinding(ratio=0.5, disease=None):
        """NEG 문장을 랜덤하게 생성: no X 또는 no finding"""
        if _R.random() < ratio:   # 확률 50% 비중 조절 가능
            return _neg_sent(disease if disease is not None else _rand_disease())
        else:
            return "There is no finding."
    
    def _make_hyb_false(class_names, pos_idx, neg_idx):
        """
        POS/NEG 정답 타입에서 사용할 '항상 거짓' HYB 문장 생성.

        가능한 패턴:
        A: There is {neg} but no {pos}   (N_not_P)
        B: There is {pos1} but no {pos2} (P_not_P)  ← pos_idx >= 2 일 때는 반드시 이 패턴
        C: There is {neg1} but no {neg2} (N_not_N)

        fallback 경로는 필요 없음 (No finding 은 상위 로직에서 이미 처리)
        """

        # 1) 양성 질환이 2개 이상이면 P_not_P 강제 생성
        if len(pos_idx) >= 2:
            p1 = _R.choice(pos_idx)
            p2_candidates = [j for j in pos_idx if j != p1]
            p2 = _R.choice(p2_candidates)
            return _hyb_sent(class_names[p1], class_names[p2])

        # 2) 양성 질환이 1개인 경우 HYB 오답 패턴 선택
        patterns = []

        # (A) N_not_P: There is {neg} but no {pos}  
        if len(neg_idx) >= 1 and len(pos_idx) >= 1:
            patterns.append("N_not_P")

        # (C) N_not_N: There is {neg1} but no {neg2}
        if len(neg_idx) >= 2:
            patterns.append("N_not_N")

        # 가능한 패턴이 있다면 반드시 존재 (no-finding 케이스는 상위에서 제외됨)
        pattern_type = _R.choice(patterns)

        if pattern_type == "N_not_P":
            n = class_names[_R.choice(neg_idx)]
            p = class_names[_R.choice(pos_idx)]
            return _hyb_sent(n, p)

        else:  # "N_not_N"
            n1 = _R.choice(neg_idx)
            n2_candidates = [j for j in neg_idx if j != n1]
            n2 = _R.choice(n2_candidates)
            return _hyb_sent(class_names[n1], class_names[n2])
        
    # ─────────────────────────── 1. No-Finding ──────────────────────────
    if labels[NF] == 1 or not pos_idx:
        # 랜덤 클래스 하나 (No Finding 제외)
        # POS / HYB 문장은 아무거나 사용해도 모두 '틀림'
        pos_sent = _pos_sent(class_names[_R.choice(range(NF))])
        neg_sent = _neg_or_nofinding()
        hyb_sent = _hyb_sent(class_names[_R.choice(range(NF))], class_names[_R.choice(range(NF))])
        choices  = [pos_sent, neg_sent, hyb_sent]
        if shuffle:
            _R.shuffle(choices)
        return choices, choices.index(neg_sent)   # 정답 = 부정문

    # ───────────────────── 2. No-Finding 아님 ──────────────────────
    # 2-1. 정답 타입 무작위
    if no_hyb:
        ans_type = _R.choice(["POS", "NEG"])
    else:
        ans_type = _R.choice(["POS", "NEG", "HYB"])

    # 2-2. 각 타입별 문장 정의
    if ans_type == "POS":
        pos_sent = _pos_sent(class_names[_R.choice(pos_idx)])              # ✔ 정답
        neg_sent = _neg_or_nofinding(disease=class_names[_R.choice(pos_idx)]) # 오답
        hyb_sent = _make_hyb_false(class_names, pos_idx, neg_idx)            # 오답
        answer  = pos_sent

    elif ans_type == "NEG":
        # 1) 정답으로 사용할 음성 질환 하나 선택
        neg_cls_idx = _R.choice(neg_idx)   # 정답 문장의 질환 인덱스 (NEG용)

        # 2) pos_sent 에 사용할 질환 인덱스를 50% 확률로 neg_cls_idx 와 같게 설정
        if len(neg_idx) == 1:
            # 음성 질환이 하나뿐이면 무조건 동일
            pos_cls_idx = neg_cls_idx
        else:
            if _R.random() < 0.75:
                # 75% 확률: 같은 질환 사용 → "There is N." vs "There is no N."
                pos_cls_idx = neg_cls_idx
            else:
                # 나머지 25%: 다른 음성 질환 사용
                other_negs = [i for i in neg_idx if i != neg_cls_idx]
                pos_cls_idx = _R.choice(other_negs)

        # 실제 문장 생성
        pos_sent = _pos_sent(class_names[pos_cls_idx])      # 오답
        neg_sent = _neg_sent(class_names[neg_cls_idx])      # ✔ 정답

        hyb_sent = _make_hyb_false(class_names, pos_idx, neg_idx)  # 오답
        answer  = neg_sent

    else:  # HYB
        pos_sent = _pos_sent(class_names[_R.choice(neg_idx)])               # 오답
        neg_sent = _neg_sent(class_names[_R.choice(pos_idx)])               # 오답
        hyb_sent = _hyb_sent(class_names[_R.choice(pos_idx)], class_names[_R.choice(neg_idx)])            # ✔ 정답
        answer  = hyb_sent

    choices = [pos_sent, neg_sent, hyb_sent]
    if shuffle:
        _R.shuffle(choices)
    return choices, choices.index(answer)

def generate_mcq3(labels: List[int], class_names: List[str], shuffle=True) -> Tuple[List[str], int]:
    NF_IDX = len(class_names) - 1
    
    # 정답 후보군을 성격에 따라 분리해서 저장
    pos_true = []  # 긍정문이면서 참 (예: 질환 있음)
    neg_true = []  # 부정문이면서 참 (예: 특정 질환 없음, No Finding)
    false_candidates = []

    if labels[NF_IDX] == 1:
        for d in class_names[:-1]:
            false_candidates.append(_pos_sent(d))
            neg_true.append(_neg_sent(d))
            
    for i, label in enumerate(labels[:-1]):
        disease = class_names[i]

        if label == 1:
            pos_true.append(_pos_sent(disease))   # 긍정 정답 후보
            false_candidates.append(_neg_sent(disease))
        else:
            neg_true.append(_neg_sent(disease))   # 부정 정답 후보
            false_candidates.append(_pos_sent(disease))

    if pos_true and _R.random() < 0.5:
        answer_text = _R.choice(pos_true)
    else:
        answer_text = _R.choice(neg_true)

    # --- 오답 및 최종 구성 ---
    false_pool = list(set(false_candidates))
    if len(false_pool) < 2:
        while len(false_pool) < 2:
            d = _R.choice(class_names[:-1])
            f_sent = _pos_sent(d) if labels[class_names.index(d)] == 0 else _neg_sent(d)
            if f_sent not in false_pool:
                false_pool.append(f_sent)
    
    wrong_choices = _R.sample(false_pool, 2)
    choices = [answer_text] + wrong_choices
    
    if shuffle:
        _R.shuffle(choices)
    
    return choices, choices.index(answer_text)

def get_positive_mask_for_prompt(
    batch_labels: torch.Tensor,
    prompt: str,
    disease_names: List[str],
    no_finding_index: int
) -> torch.Tensor:
    """
    각 프롬프트에 대해 배치 내에서 '해당하는' 샘플을 True/False mask로 반환.
    batch_labels: (B, C)  [마지막이 No Finding]
    disease_names: 길이 C 리스트 (질환 14 + "No Finding")
    """
    B, C = batch_labels.shape
    assert C == len(disease_names), "라벨 차원 수가 disease_names 길이와 맞지 않습니다."

    prompt_stripped = prompt.strip()

    # 1) "There is no finding."
    if prompt_stripped.lower() == "there is no finding.":
        pos_mask = batch_labels[:, no_finding_index] == 1
        return pos_mask

    # 2) 혼합문: "There is {질환1} but no {질환2}."
    #    → 질환1 라벨 == 1 AND 질환2 라벨 == 0 인 경우를 positive
    if prompt_stripped.startswith("There is ") and " but no " in prompt_stripped:
        # "There is " 이후 부분만 떼고 마지막 '.' 제거
        body = prompt_stripped[len("There is "):].rstrip(".").strip()
        # "{질환1} but no {질환2}" 형태
        parts = body.split(" but no ")
        if len(parts) != 2:
            # 형식이 다르면 매칭 안 되는 것으로 처리
            return torch.zeros(B, dtype=torch.bool, device=batch_labels.device)

        disease_name_pos = parts[0].strip()  # 질환1
        disease_name_neg = parts[1].strip()  # 질환2

        # 질환명이 리스트에 없으면 매칭 없음
        if (disease_name_pos not in disease_names) or (disease_name_neg not in disease_names):
            return torch.zeros(B, dtype=torch.bool, device=batch_labels.device)

        idx_pos = disease_names.index(disease_name_pos)
        idx_neg = disease_names.index(disease_name_neg)

        # 질환1은 1, 질환2는 0인 경우만 positive
        pos_mask = (batch_labels[:, idx_pos] == 1) & (batch_labels[:, idx_neg] == 0)
        return pos_mask

    # 3) 단순 부정문 / 긍정문
    if prompt_stripped.startswith("There is no "):
        disease_name = prompt_stripped[len("There is no "):].strip().rstrip(".")
        is_negation = True
    elif prompt_stripped.startswith("There is "):
        disease_name = prompt_stripped[len("There is "):].strip().rstrip(".")
        is_negation = False
    else:
        # 정의되지 않은 패턴
        return torch.zeros(B, dtype=torch.bool, device=batch_labels.device)

    # 질환 인덱스 찾기
    if disease_name not in disease_names:
        return torch.zeros(B, dtype=torch.bool, device=batch_labels.device)

    disease_idx = disease_names.index(disease_name)

    if not is_negation:
        # "There is {질환}" → 해당 질환 라벨이 1인 경우
        pos_mask = batch_labels[:, disease_idx] == 1
    else:
        # "There is no {질환}" → 해당 질환 라벨이 0인 경우
        pos_mask = batch_labels[:, disease_idx] == 0

    return pos_mask

def sample_pos_neg_for_all_prompts(
    batch_labels: torch.Tensor,
    prompts: List[str],
    disease_names: List[str],
    no_finding_index: int,
    num_negatives: int = 3,
) -> Dict[str, Any]:
    """
    배치 레이블과 프롬프트 리스트를 기반으로, 각 프롬프트에 대해
    (idx_list, ans_idx)를 반환하되,

    - 배치 내 인덱스들이 가능한 한 균등하게 사용되도록 샘플링
    - 혼합문("There is A but no B.")의 경우:
        * pos: A=1, B=0
        * hard negative: A=1, B=1
        * 배치 내 hard negative가 1개도 없으면 아예 스킵 (딕셔너리에 넣지 않음)
    """

    B = batch_labels.shape[0]
    device = batch_labels.device

    # 각 이미지 인덱스의 사용 횟수
    usage = [0] * B

    out_dict: Dict[str, Any] = {}

    for prompt in prompts:
        prompt_stripped = prompt.strip()

        # 1) pos / neg mask 구하기
        pos_mask = get_positive_mask_for_prompt(batch_labels, prompt, disease_names, no_finding_index)
        pos_mask = pos_mask.to(device=device)
        neg_mask = ~pos_mask

        pos_indices = pos_mask.nonzero(as_tuple=False).flatten().tolist()
        neg_indices = neg_mask.nonzero(as_tuple=False).flatten().tolist()

        # (1) pos 없으면 스킵
        if len(pos_indices) == 0:
            continue

        # (2) neg가 필요한 수보다 적으면 스킵
        if len(neg_indices) < num_negatives:
            continue

        # 2) pos_idx 선택: usage가 가장 적은 인덱스들 중에서 선택
        min_pos_usage = min(usage[i] for i in pos_indices)
        candidate_pos = [i for i in pos_indices if usage[i] == min_pos_usage]
        pos_idx = random.choice(candidate_pos)

        # -------------------------------
        # 3) 혼합문이면 hard negative 후보 계산
        #    "There is A but no B."
        # -------------------------------
        hard_neg_indices: List[int] = []
        is_mixed = False

        if prompt_stripped.startswith("There is ") and " but no " in prompt_stripped:
            body = prompt_stripped[len("There is "):].rstrip(".").strip()
            parts = body.split(" but no ")
            if len(parts) == 2:
                disease_A = parts[0].strip()
                disease_B = parts[1].strip()
                if disease_A in disease_names and disease_B in disease_names:
                    idx_A = disease_names.index(disease_A)
                    idx_B = disease_names.index(disease_B)

                    # hard negative: A=1, B=1 이면서 현재 prompt 기준 neg인 샘플
                    hard_mask = (batch_labels[:, idx_A] == 1) & (batch_labels[:, idx_B] == 1)
                    hard_mask = hard_mask & neg_mask.to(device=batch_labels.device)
                    hard_neg_indices = hard_mask.nonzero(as_tuple=False).flatten().tolist()
                    is_mixed = True

        # 혼합문인데 hard negative가 하나도 없다면 이 prompt는 사용하지 않음
        if is_mixed and len(hard_neg_indices) == 0:
            continue

        # -------------------------------
        # 4) neg 샘플링 (hard negative 우선)
        # -------------------------------
        sampled_negs: List[int] = []

        if is_mixed:
            # hard_neg_indices는 이미 len > 0 이라고 보장됨
            hard_neg_sorted = sorted(
                hard_neg_indices,
                key=lambda i: (usage[i], random.random())
            )

            # 우선 hard negative에서 뽑을 수 있는 만큼 뽑기
            num_from_hard = min(num_negatives, len(hard_neg_sorted))
            selected_hards = hard_neg_sorted[:num_from_hard]
            sampled_negs.extend(selected_hards)

            # 나머지는 일반 neg에서 usage 기준으로 추가
            remaining = num_negatives - num_from_hard
            if remaining > 0:
                hard_set = set(hard_neg_indices)
                easy_neg_indices = [i for i in neg_indices if i not in hard_set]

                easy_neg_sorted = sorted(
                    easy_neg_indices,
                    key=lambda i: (usage[i], random.random())
                )

                if len(easy_neg_sorted) < remaining:
                    # 이론상 len(neg_indices) >= num_negatives 이므로 거의 안 걸림
                    sampled_negs.extend(easy_neg_sorted)
                else:
                    sampled_negs.extend(easy_neg_sorted[:remaining])
        else:
            # 일반문 (혹은 혼합문이 아닌 경우): 기존 balanced neg 샘플링
            neg_sorted = sorted(
                neg_indices,
                key=lambda i: (usage[i], random.random())
            )
            sampled_negs = neg_sorted[:num_negatives]

        # 안전하게 보정: 혹시라도 길이가 num_negatives보다 작으면 스킵
        if len(sampled_negs) < num_negatives:
            continue

        # -------------------------------
        # 5) pos + neg 합치고 shuffle + 정답 위치(ans_idx) 계산
        # -------------------------------
        idx_list = [pos_idx] + sampled_negs
        random.shuffle(idx_list)
        ans_idx = idx_list.index(pos_idx)

        # 6) usage 업데이트
        for i in idx_list:
            usage[i] += 1

        # 7) 결과 저장 (prompt를 key로)
        out_dict[prompt] = (idx_list, ans_idx)

    return out_dict

def extract_disease_indices_from_prompt(
    prompt: str,
    disease_names: List[str],
) -> Set[int]:
    """
    주어진 프롬프트 문자열에서 disease_names에 포함된 질환명이
    몇 개 등장하는지 찾아, 그 인덱스 집합을 반환.
    
    - "There is A but no B." 패턴을 우선적으로 처리
    - 그 외에는 단순 substring 매칭으로 disease_names를 탐색
    """
    prompt_stripped = prompt.strip()
    found: Set[int] = set()

    # 1) "There is A but no B." 패턴 우선 처리
    if prompt_stripped.startswith("There is ") and " but no " in prompt_stripped:
        body = prompt_stripped[len("There is "):].rstrip(".").strip()
        parts = body.split(" but no ")
        if len(parts) == 2:
            disease_A = parts[0].strip()
            disease_B = parts[1].strip()
            if disease_A in disease_names:
                found.add(disease_names.index(disease_A))
            if disease_B in disease_names:
                found.add(disease_names.index(disease_B))

    # 2) 일반적인 substring 매칭 (중복 추가 방지)
    lowered_prompt = prompt_stripped.lower()
    for idx, name in enumerate(disease_names):
        if name.lower() in lowered_prompt:
            found.add(idx)

    return found

def select_prompts_max_disease_coverage(
    mcq_dict: Dict[str, Any],
    disease_names: List[str],
    B: int,
) -> Dict[str, Any]:
    """
    sample_pos_neg_for_all_prompts 로 생성된 mcq_dict (Q개) 중에서
    최대 B개의 프롬프트를 선택하되,
    선택된 프롬프트들에 등장하는 질환 인덱스의 coverage(서로 다른 질환 수)가
    최대가 되도록 하는 greedy 알고리즘.

    반환: 선택된 프롬프트만 남긴 새로운 mcq_dict 서브셋
    """

    # 0) 입력 프롬프트/항목을 리스트로 정렬
    items: List[Tuple[str, Any]] = list(mcq_dict.items())
    Q = len(items)
    if B >= Q:
        # B가 Q 이상이면 전체 사용
        return mcq_dict

    prompts: List[str] = [p for p, _ in items]

    # 1) 각 프롬프트별로 어떤 질환 인덱스들을 커버하는지 미리 계산
    diseases_per_prompt: List[Set[int]] = [
        extract_disease_indices_from_prompt(p, disease_names)
        for p in prompts
    ]

    # 2) 탐욕적 선택을 위한 상태 변수
    selected_indices: List[int] = []
    covered_diseases: Set[int] = set()
    remaining_indices: Set[int] = set(range(Q))

    # 3) 최대 B개까지 반복해서 선택
    for _ in range(B):
        best_gain = -1
        best_candidates: List[int] = []

        for i in remaining_indices:
            # 현재 프롬프트 i를 선택했을 때 새로 커버되는 질환 수
            new_coverage = diseases_per_prompt[i] - covered_diseases
            gain = len(new_coverage)

            if gain > best_gain:
                best_gain = gain
                best_candidates = [i]
            elif gain == best_gain:
                best_candidates.append(i)

        if best_gain <= 0:
            # 더 이상 새로운 질환을 커버할 수 있는 프롬프트가 없음
            # 남은 것 중에서 아무거나(혹은 heuristics 기반) 채워 넣기
            if not remaining_indices:
                break
            # 예: 질환 수가 더 많은 프롬프트 위주로 선택
            remaining_list = list(remaining_indices)
            remaining_list.sort(
                key=lambda i: len(diseases_per_prompt[i]),
                reverse=True,
            )
            chosen = remaining_list[0]
        else:
            # best_gain이 양수인 프롬프트들 중에서 랜덤하게 하나 선택 (tie-breaking)
            chosen = random.choice(best_candidates)

        # 선택한 프롬프트로 상태 업데이트
        selected_indices.append(chosen)
        covered_diseases |= diseases_per_prompt[chosen]
        remaining_indices.remove(chosen)

        if not remaining_indices:
            break

    # 4) 선택된 인덱스들만 모아서 새로운 mcq_dict 생성
    selected_prompts = {prompts[i]: items[i][1] for i in selected_indices}

    return selected_prompts


def build_t2i_mcq_batch(
    batch: Dict[str, Any],
    tokenizer,
    prompts: List[str],
    class_names: List[str],
    max_length: int,
    num_negatives: int = 2,
    no_hyb: bool = False
) -> Dict[str, Any]:
    
    imgs: torch.Tensor = batch["imgs"]              # (B, C, H, W)
    B, C, H, W = imgs.shape
    
    if no_hyb :
        prompts = [p for p in prompts if " but no " not in p]
    
    mcq_dict = sample_pos_neg_for_all_prompts(
        batch_labels=batch["label"],
        prompts=prompts,
        disease_names=class_names,
        no_finding_index=14,
        num_negatives=num_negatives
    )
    
    mcq_dict = select_prompts_max_disease_coverage(
    mcq_dict=mcq_dict,
    disease_names=class_names,
    B=B)
    
    # 1) dict에서 prompt, idx_list, ans_idx를 순서대로 뽑기
    prompts: List[str] = []
    all_idx_lists: List[List[int]] = []
    all_ans_indices: List[int] = []

    for prompt, (idx_list, ans_idx) in mcq_dict.items():
        if len(idx_list) == 0:
            continue
        prompts.append(prompt)
        all_idx_lists.append(idx_list)
        all_ans_indices.append(ans_idx)

    if len(prompts) == 0:
        # 사용할 MCQ가 없으면 그대로 반환
        return batch
    
    Q = len(prompts)          # MCQ 개수
    N = len(all_idx_lists[0]) # 보기 개수 (모든 prompt에서 동일하다고 가정)

    # 2) 이미지 인덱스 텐서: (Q, N)
    idx_tensor = torch.tensor(all_idx_lists, dtype=torch.long)  # (Q, N)

    # 3) 실제 이미지 gather: (Q, N, C, H, W)
    flat_indices = idx_tensor.view(-1)     # (Q*N,)
    flat_imgs = imgs[flat_indices]         # (Q*N, C, H, W)
    mcq_imgs = flat_imgs.view(Q, N, C, H, W)

    # 4) 프롬프트 토큰화: (Q, L)
    tok = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = tok["input_ids"]
    attention_mask = tok["attention_mask"]
    token_type_ids = tok.get("token_type_ids", None)

    # 5) 텍스트를 보기 수 N만큼 repeat → (Q, N, L)
    input_ids = input_ids.unsqueeze(1).expand(Q, N, -1)           # (Q, N, L)
    attention_mask = attention_mask.unsqueeze(1).expand(Q, N, -1) # (Q, N, L)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.unsqueeze(1).expand(Q, N, -1)  # (Q, N, L)

    # 6) 정답 인덱스: (Q,)
    answers = torch.tensor(all_ans_indices, dtype=torch.long)

    # 7) 기존 batch에 MCQ용 텐서들을 추가해서 반환
    batch_out = dict(batch)  # 원본은 건드리지 않고 shallow copy

    batch_out["imgs"] = mcq_imgs                 # (Q, N, C, H, W)
    batch_out["caption_ids"] = input_ids             # (Q, N, L)
    batch_out["attention_mask"] = attention_mask   # (Q, N, L)
    batch_out["token_type_ids"] = token_type_ids   # (Q, N, L) 또는 None
    batch_out["answer_idx"] = answers                 # (Q,)
    
    return batch_out

def _sp(s: str) -> str:
    return s.replace("_", " ")

def _pick_k(rng: random.Random, pool: List[int], k: int) -> List[int]:
    k = max(0, min(k, len(pool)))
    return rng.sample(pool, k)

# def generate_prompt(
#     labels: List[int],
#     class_names: List[str],
#     n_findings_range: Tuple[int, int] = (2, 3),
#     no_finding_name: str = "No Finding",
#     seed: Optional[int] = None,
#     p_multi: float = 0.45,  # 한 문장에 2–3개 질환을 묶을 확률
# ) -> str:
#     """
#     멀티라벨 벡터를 3–4개의 완전한 문장으로 verbalize한 단일 프롬프트를 생성합니다.
#     - seed=None이면 매 호출마다 다른 결과(고엔트로피 시드).
#     - NF 양성이면 "There is no finding."을 반드시 포함.
#     - 한 문장에 2–3개의 질환을 묶는 복수 템플릿을 확률적으로 사용.
#     - 동일 문장(문자열)과 동일 조합(양/음성, 질환 set)이 중복 생성되지 않도록 보장.
#     """

#     assert len(labels) == len(class_names), "labels/class_names length mismatch"

#     # --- 고엔트로피 시드 설정 ---
#     if seed is None:
#         sysrand = random.SystemRandom()
#         seed = sysrand.getrandbits(64)
#     rng = random.Random(seed)

#     # --- 인덱스 분리 ---
#     try:
#         nf_idx = class_names.index(no_finding_name)
#     except ValueError:
#         nf_idx = len(class_names) - 1  # fallback

#     disease_idx = [i for i in range(len(class_names)) if i != nf_idx]
#     pos_idx = [i for i in disease_idx if labels[i] == 1]
#     neg_idx = [i for i in disease_idx if labels[i] == 0]
#     has_nf = (0 <= nf_idx < len(labels)) and (labels[nf_idx] == 1)

#     # --- 문장 수 ---
#     n_min, n_max = n_findings_range
#     n_target = rng.randint(n_min, n_max)

#     # --- 템플릿 ---
#     POS_SINGLE_POOL = [
#         "{x} is present.",
#         "{x} is noted.",
#         "There is {x}.",
#         "There is evidence of {x}.",
#         "Findings are consistent with {x}.",
#     ]
#     POS_MULTI_POOL = [
#         "{xs} are present.",
#         "There are {xs}.",
#         "Findings are consistent with {xs}.",
#         "There is evidence of {xs}.",
#     ]
#     NEG_SINGLE_POOL = [
#         "No {x} is identified.",
#         "No {x} is seen.",
#         "{x} is not present.",
#         "{x} is absent.",
#         "No radiographic evidence of {x}.",
#         "There is no evidence of {x}.",
#     ]
#     NEG_MULTI_POOL = [
#         "No {xs} are identified.",
#         "No {xs} are seen.",
#         "No radiographic evidence of {xs}.",
#         "There is no evidence of {xs}.",
#     ]

#     # 템플릿을 '중복 없이' 소모하기 위한 상태
#     pos_single_tpls = POS_SINGLE_POOL[:]
#     pos_multi_tpls  = POS_MULTI_POOL[:]
#     neg_single_tpls = NEG_SINGLE_POOL[:]
#     neg_multi_tpls  = NEG_MULTI_POOL[:]
#     rng.shuffle(pos_single_tpls)
#     rng.shuffle(pos_multi_tpls)
#     rng.shuffle(neg_single_tpls)
#     rng.shuffle(neg_multi_tpls)

#     def _choose_tpl(pool: List[str], backup: List[str]) -> str:
#         """가능하면 pool에서 중복 없이 pop, 고갈 시 backup 셔플 후 재공급."""
#         nonlocal rng
#         if not pool:
#             pool.extend(backup)
#             rng.shuffle(pool)
#         return pool.pop()

#     def _fmt_pos(name: str) -> str:
#         tpl = _choose_tpl(pos_single_tpls, POS_SINGLE_POOL)
#         return tpl.format(x=_sp(name))

#     def _fmt_pos_multi(names: List[str]) -> str:
#         xs = _oxford_join([_sp(n) for n in names])
#         tpl = _choose_tpl(pos_multi_tpls, POS_MULTI_POOL)
#         return tpl.format(xs=xs)

#     def _fmt_neg(name: str) -> str:
#         tpl = _choose_tpl(neg_single_tpls, NEG_SINGLE_POOL)
#         return tpl.format(x=_sp(name))

#     def _fmt_neg_multi(names: List[str]) -> str:
#         xs = _oxford_join([_sp(n) for n in names])
#         tpl = _choose_tpl(neg_multi_tpls, NEG_MULTI_POOL)
#         return tpl.format(xs=xs)

#     # --- 중복 방지 상태 ---
#     used_sentences = set()  # 문자열 기준 완전 중복 방지
#     used_groups = set()     # ('pos'|'neg', frozenset({names...})) 조합 중복 방지

#     def _try_add_sentence(s: str, kind: str, names: List[str]) -> bool:
#         """문장/조합 중복을 검사하고 통과하면 등록."""
#         key_sent = s.strip().lower()
#         key_group = (kind, frozenset(n.strip().lower() for n in names))
#         if key_sent in used_sentences or key_group in used_groups:
#             return False
#         used_sentences.add(key_sent)
#         used_groups.add(key_group)
#         sentences.append(s)
#         return True

#     def _build_sentences_from_names(pos_names: List[str], neg_names: List[str], target: int) -> None:
#         """
#         pos_names, neg_names 풀에서 target 개수 문장을 생성.
#         - p_multi 확률로 2–3개를 묶어 복수 문장 생성
#         - 각 생성 시 중복(문장/조합) 검사를 통과한 경우만 채택
#         - 제한된 재시도 후에도 실패하면 fallback(단수/다른 템플릿)로 채움
#         """
#         P = pos_names[:]
#         N = neg_names[:]
#         rng.shuffle(P)
#         rng.shuffle(N)

#         max_attempts = 10 * target  # 안전 여유

#         attempts = 0
#         while len(sentences) < target and (P or N) and attempts < max_attempts:
#             attempts += 1
#             # 양성/음성 우선 랜덤
#             choose_pos = (rng.random() < 0.5 and P) or not N
#             if choose_pos and P:
#                 # 복수 시도
#                 if rng.random() < p_multi and len(P) >= 2:
#                     gsize = 2 + int(rng.random() < 0.35 and len(P) >= 3)
#                     group = [P.pop() for _ in range(gsize)]
#                     cand = _fmt_pos_multi(group)
#                     if not _try_add_sentence(cand, "pos", group):
#                         # 실패 시 되돌리고 단수로 시도
#                         for n in group: P.append(n)
#                         rng.shuffle(P)
#                         single = P.pop()
#                         cand = _fmt_pos(single)
#                         if not _try_add_sentence(cand, "pos", [single]):
#                             # 실패면 다른 템플릿으로 재시도 유도
#                             P.insert(0, single)
#                             continue
#                 else:
#                     single = P.pop()
#                     cand = _fmt_pos(single)
#                     if not _try_add_sentence(cand, "pos", [single]):
#                         # 실패 시 다른 템플릿으로 다시 시도
#                         P.insert(0, single)
#                         continue
#             else:
#                 if N:
#                     if rng.random() < p_multi and len(N) >= 2:
#                         gsize = 2 + int(rng.random() < 0.35 and len(N) >= 3)
#                         group = [N.pop() for _ in range(gsize)]
#                         cand = _fmt_neg_multi(group)
#                         if not _try_add_sentence(cand, "neg", group):
#                             for n in group: N.append(n)
#                             rng.shuffle(N)
#                             single = N.pop()
#                             cand = _fmt_neg(single)
#                             if not _try_add_sentence(cand, "neg", [single]):
#                                 N.insert(0, single)
#                                 continue
#                     else:
#                         single = N.pop()
#                         cand = _fmt_neg(single)
#                         if not _try_add_sentence(cand, "neg", [single]):
#                             N.insert(0, single)
#                             continue
#                 elif P:
#                     # 음성 고갈 시 양성으로 대체
#                     single = P.pop()
#                     cand = _fmt_pos(single)
#                     if not _try_add_sentence(cand, "pos", [single]):
#                         P.insert(0, single)
#                         continue

#         # 부족하면 남은 풀로 보충(단수 위주, 중복 필터 유지)
#         filler_attempts = 0
#         while len(sentences) < target and filler_attempts < 5 * target:
#             filler_attempts += 1
#             if neg_names:
#                 # 음성 우선
#                 name = rng.choice(neg_names)
#                 cand = _fmt_neg(name)
#                 if _try_add_sentence(cand, "neg", [name]):
#                     continue
#             if pos_names:
#                 name = rng.choice(pos_names)
#                 cand = _fmt_pos(name)
#                 if _try_add_sentence(cand, "pos", [name]):
#                     continue
#             break  # 더 이상 채울 수 없음

#     sentences: List[str] = []

#     # ---------------- NF branch ----------------
#     if has_nf or len(pos_idx) == 0:
#         # NF 강제 포함
#         base = "There is no finding."
#         used_sentences.add(base.strip().lower())
#         used_groups.add(("neg", frozenset({"__no_finding__"})))  # 더미 그룹키
#         sentences.append(base)

#         neg_pool_names = [_sp(class_names[i]) for i in neg_idx] if neg_idx else [_sp(class_names[i]) for i in disease_idx]
#         rng.shuffle(neg_pool_names)

#         need = max(n_target - 1, 0)
#         _build_sentences_from_names(pos_names=[], neg_names=neg_pool_names, target=need)
#         rng.shuffle(sentences)  # NF 위치는 고정할 필요 없음(포함만 보장)
#         return " ".join(sentences).replace(".  ", ". ").strip()

#     # ---------------- positive branch ----------------
#     pos_take_idx = pos_idx[:]                     # ← 전체 양성 질환 유지
#     rng.shuffle(pos_take_idx)
#     pos_names = [_sp(class_names[i]) for i in pos_take_idx]

#     # n_target 이 pos 개수보다 작은 경우 → pos 개수로 자동 상향 조절
#     if n_target < len(pos_names):
#         n_target = len(pos_names)
#     pos_names = [_sp(class_names[i]) for i in pos_take_idx]

#     remain_neg_idx = [i for i in neg_idx if i not in pos_take_idx]
#     # 음성 후보는 여유 있게 준비
#     n_neg_need = max(n_target, 2)
#     neg_take_idx = _pick_k(rng, remain_neg_idx, min(len(remain_neg_idx), n_neg_need))
#     neg_names = [_sp(class_names[i]) for i in neg_take_idx]

#     _build_sentences_from_names(pos_names, neg_names, n_target)

#     # 혹시 과다 생성되었으면 랜덤 컷 (일반적으로 정확히 맞춰짐)
#     if len(sentences) > n_target:
#         rng.shuffle(sentences)
#         sentences = sentences[:n_target]

#     rng.shuffle(sentences)
#     return " ".join(sentences).replace(".  ", ". ").strip()

def generate_prompt(
    labels: List[int],
    class_names: List[str],
    n_findings_range: Tuple[int, int] = (2, 3),
    no_finding_name: str = "No Finding",
    seed: Optional[int] = None,
    p_multi: float = 0.45,  # 호환성 유지를 위해 남겨두지만 사용하지 않습니다.
) -> str:
    """
    멀티라벨 벡터를 2–3개의 '단수' 문장으로만 verbalize한 단일 프롬프트를 생성합니다.
    - NF 양성이면 "There is no finding."을 반드시 포함하고 나머지는 음성 단수 문장으로 채움.
    - 각 문장과 (양/음성, 질환) 조합의 중복을 방지.
    - 복수(2–3개 묶음) 템플릿/로직은 제거되었습니다.
    """

    assert len(labels) == len(class_names), "labels/class_names length mismatch"

    # --- 고엔트로피 시드 설정 ---
    if seed is None:
        sysrand = random.SystemRandom()
        seed = sysrand.getrandbits(64)
    rng = random.Random(seed)

    # --- 인덱스 분리 ---
    try:
        nf_idx = class_names.index(no_finding_name)
    except ValueError:
        nf_idx = len(class_names) - 1  # fallback

    disease_idx = [i for i in range(len(class_names)) if i != nf_idx]
    pos_idx = [i for i in disease_idx if labels[i] == 1]
    neg_idx = [i for i in disease_idx if labels[i] == 0]
    has_nf = (0 <= nf_idx < len(labels)) and (labels[nf_idx] == 1)

    # --- 문장 수 ---
    n_min, n_max = n_findings_range
    n_target = rng.randint(n_min, n_max)

    # --- 단수 템플릿 ---
    POS_SINGLE_POOL = [
        "{x} is present.",
        "{x} is noted.",
        "There is {x}.",
        "There is evidence of {x}.",
        "Findings are consistent with {x}.",
    ]
    NEG_SINGLE_POOL = [
        "No {x} is identified.",
        "No {x} is seen.",
        "{x} is not present.",
        "{x} is absent.",
        "No radiographic evidence of {x}.",
        "There is no evidence of {x}.",
    ]

    # 템플릿을 '중복 없이' 소모하기 위한 상태
    pos_single_tpls = POS_SINGLE_POOL[:]
    neg_single_tpls = NEG_SINGLE_POOL[:]
    rng.shuffle(pos_single_tpls)
    rng.shuffle(neg_single_tpls)

    def _choose_tpl(pool: List[str], backup: List[str]) -> str:
        """가능하면 pool에서 중복 없이 pop, 고갈 시 backup 셔플 후 재공급."""
        nonlocal rng
        if not pool:
            pool.extend(backup)
            rng.shuffle(pool)
        return pool.pop()

    def _fmt_pos(name: str) -> str:
        tpl = _choose_tpl(pos_single_tpls, POS_SINGLE_POOL)
        return tpl.format(x=_sp(name))

    def _fmt_neg(name: str) -> str:
        tpl = _choose_tpl(neg_single_tpls, NEG_SINGLE_POOL)
        return tpl.format(x=_sp(name))

    # --- 중복 방지 상태 ---
    used_sentences = set()  # 문자열 기준 완전 중복 방지
    used_groups = set()     # ('pos'|'neg', frozenset({name})) 조합 중복 방지

    def _try_add_sentence(s: str, kind: str, name: str) -> bool:
        """문장/조합 중복을 검사하고 통과하면 등록."""
        key_sent = s.strip().lower()
        key_group = (kind, frozenset({name.strip().lower()}))
        if key_sent in used_sentences or key_group in used_groups:
            return False
        used_sentences.add(key_sent)
        used_groups.add(key_group)
        sentences.append(s)
        return True

    sentences: List[str] = []

    # ---------------- NF branch ----------------
    if has_nf or len(pos_idx) == 0:
        # NF 강제 포함
        base = "There is no finding."
        used_sentences.add(base.strip().lower())
        used_groups.add(("neg", frozenset({"__no_finding__"})))  # 더미 그룹키
        sentences.append(base)

        # 남은 문장들을 '단수 음성 문장'으로 채움
        neg_pool_names = [_sp(class_names[i]) for i in neg_idx] if neg_idx else [_sp(class_names[i]) for i in disease_idx]
        rng.shuffle(neg_pool_names)

        need = max(n_target - 1, 0)
        attempts = 0
        while len(sentences) < 1 + need and attempts < 10 * max(1, need):
            attempts += 1
            if not neg_pool_names:
                break
            name = rng.choice(neg_pool_names)
            cand = _fmt_neg(name)
            _try_add_sentence(cand, "neg", name)

        rng.shuffle(sentences)
        return " ".join(sentences).replace(".  ", ". ").strip()

    # ---------------- positive branch ----------------
    # 모든 양성 질환을 최소 1회 단수 문장으로 기술
    pos_take_idx = pos_idx[:]
    rng.shuffle(pos_take_idx)
    pos_names = [_sp(class_names[i]) for i in pos_take_idx]

    # n_target이 pos 개수보다 작으면 pos 개수에 맞춰 상향 (각 양성 최소 1회 보장)
    if n_target < len(pos_names):
        n_target = len(pos_names)

    # 1) 양성 단수 문장 채우기
    for name in pos_names:
        cand = _fmt_pos(name)
        _try_add_sentence(cand, "pos", name)

    # 2) 남는 분량은 음성 단수 문장으로 보충
    remain_neg_idx = [i for i in neg_idx if i not in pos_take_idx]
    n_neg_need = max(n_target - len(sentences), 0)
    neg_take_idx = _pick_k(rng, remain_neg_idx, min(len(remain_neg_idx), max(n_neg_need, 2)))
    neg_names = [_sp(class_names[i]) for i in neg_take_idx]
    rng.shuffle(neg_names)

    attempts = 0
    while len(sentences) < n_target and attempts < 10 * max(1, n_neg_need):
        attempts += 1        # 단수 음성 문장만 시도
        if not neg_names:
            break
        name = rng.choice(neg_names)
        cand = _fmt_neg(name)
        _try_add_sentence(cand, "neg", name)

    # 혹시 과다 생성되었으면 랜덤 컷
    if len(sentences) > n_target:
        rng.shuffle(sentences)
        sentences = sentences[:n_target]

    rng.shuffle(sentences)
    return " ".join(sentences).replace(".  ", ". ").strip()

class NIHMCQOnlyDataset(Dataset):
    def __init__(self, df, cfg, transform):
        self.df = df
        self.cfg = cfg
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)

        self.class_names = ['Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Pulmonary Infiltration', 'Pulmonary Mass', 'Lung Nodule', 'Pneumonia', 'Pneumothorax',
            'Pulmonary Consolidation', 'Pulmonary Edema', 'Pulmonary Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia', 'no finding']
        self.pos_prompts = {cls: f"There is {cls.replace('_', ' ')}." for cls in self.class_names}
        self.neg_prompts = {cls: f"There is no {cls.replace('_', ' ')}." for cls in self.class_names[:-1]}

    def __len__(self):
        return len(self.df)

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )
        return resized_img
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(row['Path']), 0)
        img = self._resize_img(img, self.cfg.data.image.imsize)
        img = Image.fromarray(img).convert("RGB")
        img = self.transform(img)

        labels = row.iloc[2:].tolist()
        
        choices, answer_idx = generate_mcq2(labels, self.class_names) if self.cfg.data.text.generate_mcq2 else generate_mcq(labels, self.class_names, no_hyb=self.cfg.data.text.no_hyb)
        
        tokens = self.tokenizer( # (4, seq_len)
            choices,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.data.text.word_num
        )
        cap_len = torch.tensor(
            [int((ids != 0).sum()) for ids in tokens["input_ids"]],
            dtype=torch.long
        )

        return {
            "imgs": img,
            "caption_ids" : tokens["input_ids"],
            "attention_mask" : tokens["attention_mask"],
            "token_type_ids" : tokens["token_type_ids"],
            "cap_len" : cap_len,
            "label" : torch.tensor(labels, dtype=torch.float),
            "answer_idx": torch.tensor(answer_idx, dtype=torch.long),
        }
        
def get_counter_prompt(answer: str, class_names) -> str:
    # Case 1: No Finding 문구인 경우 -> 랜덤 질환 긍정문
    if "no finding" in answer.lower():
        return _pos_sent(_R.choice(class_names[:-1]))
    
    # Case 2: 부정문인 경우 (no/not 포함) -> 긍정문으로 전환
    if "no " in answer.lower() or "not " in answer.lower():
        for d in class_names[:-1]:
            if d.lower() in answer.lower(): return _pos_sent(d)
        return "There is a finding." # 매칭 실패시 기본 긍정형
        
    # Case 3: 긍정문인 경우 -> 부정문으로 전환
    for d in class_names[:-1]:
        if d.lower() in answer.lower(): return _neg_sent(d)
    return "There is no finding." # 매칭 실패시 기본 부정형

class NIHMCQEDLDataset(Dataset):
    def __init__(self, df, cfg, transform):
        self.df = df
        self.cfg = cfg
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)

        self.class_names = ['Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Pulmonary Infiltration', 'Pulmonary Mass', 'Lung Nodule', 'Pneumonia', 'Pneumothorax',
            'Pulmonary Consolidation', 'Pulmonary Edema', 'Pulmonary Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia', 'no finding']
        self.pos_prompts = {cls: f"There is {cls.replace('_', ' ')}." for cls in self.class_names}
        self.neg_prompts = {cls: f"There is no {cls.replace('_', ' ')}." for cls in self.class_names[:-1]}

    def __len__(self):
        return len(self.df)

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )
        return resized_img
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(row['Path']), 0)
        img = self._resize_img(img, self.cfg.data.image.imsize)
        img = Image.fromarray(img).convert("RGB")
        img = self.transform(img)

        labels = row.iloc[2:].tolist()
        
        choices, answer_idx = generate_mcq3(labels, self.class_names)
        
        answer_prompt = choices[answer_idx]
        
        counter_prompt = get_counter_prompt(answer_prompt, self.class_names)
    
        choices.append(counter_prompt)   
        
        tokens = self.tokenizer( # (4, seq_len)
            choices,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.data.text.word_num
        )
        cap_len = torch.tensor(
            [int((ids != 0).sum()) for ids in tokens["input_ids"]],
            dtype=torch.long
        )

        return {
            "imgs": img,
            "caption_ids" : tokens["input_ids"],
            "attention_mask" : tokens["attention_mask"],
            "token_type_ids" : tokens["token_type_ids"],
            "cap_len" : cap_len,
            "label" : torch.tensor(labels, dtype=torch.float),
            "answer_idx": torch.tensor(answer_idx, dtype=torch.long),
        }

class NIHMCQOnly2Dataset(Dataset):
    def __init__(self, root, cfg, transform):
        self.root = root
        self.cfg = cfg
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)

        self.df = pd.read_csv(os.path.join(root, 'Data_Entry_2017.csv'))
        self.df['path'] = self.df['Image Index'].map(self._build_path_map())

        self.class_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding']
        self.pos_prompts = {cls: f"There is {cls.replace('_', ' ')}." for cls in self.class_names}
        self.neg_prompts = {cls: f"There is no {cls.replace('_', ' ')}." for cls in self.class_names[:-1]}
        
    def _build_path_map(self):
        paths = glob(os.path.join(self.root, 'images*', '*', '*.png'))
        return {os.path.basename(p): p for p in paths}

    def __len__(self):
        return len(self.df)

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )
        return resized_img
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(row['path']), 0)
        img = self._resize_img(img, self.cfg.data.image.imsize)
        img = Image.fromarray(img).convert("RGB")
        img = self.transform(img)

        label_str = row['Finding Labels'].split('|')
        labels = [1 if cls in label_str else 0 for cls in self.class_names]
        
        choices, answer_idx = generate_mcq2(labels, self.class_names)
        
        tokens = self.tokenizer( # (4, seq_len)
            choices,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.data.text.word_num
        )
        cap_len = torch.tensor(
            [int((ids != 0).sum()) for ids in tokens["input_ids"]],
            dtype=torch.long
        )

        return {
            "imgs": img,
            "caption_ids" : tokens["input_ids"],
            "attention_mask" : tokens["attention_mask"],
            "token_type_ids" : tokens["token_type_ids"],
            "cap_len" : cap_len,
            "label" : torch.tensor(labels, dtype=torch.float),
            "answer_idx": torch.tensor(answer_idx, dtype=torch.long),
        }
        
class NIHMCQ2Dataset(Dataset):
    def __init__(self, root, cfg, transform):
        self.root = root
        self.cfg = cfg
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)

        self.df = pd.read_csv(os.path.join(root, 'Data_Entry_2017.csv'))
        self.df['Finding Labels'] = self.df['Finding Labels'].str.replace('_', ' ', regex=False)
        self.df['path'] = self.df['Image Index'].map(self._build_path_map())

        self.class_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia', 'No Finding']
        
        self.prompts = {cls: f"There is {cls.replace('_', ' ')}." for cls in self.class_names}
        self.neg_prompts = {cls: f"There is no {cls.replace('_', ' ')}." for cls in self.class_names[:-1]}
        
    def _build_path_map(self):
        paths = glob(os.path.join(self.root, 'images*', '*', '*.png'))
        return {os.path.basename(p): p for p in paths}

    def __len__(self):
        return len(self.df)

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(row['path']), 0)
        img = self._resize_img(img, self.cfg.data.image.imsize)
        img = Image.fromarray(img).convert("RGB")
        img = self.transform(img)

        label_str = row['Finding Labels'].split('|')
        labels = [1 if cls in label_str else 0 for cls in self.class_names]
        
        choices, answer_idx = generate_mcq(labels, self.class_names)
        # print(f"True Labels: {[cls for cls, label in zip(self.class_names, labels) if label == 1]}")
        # print(f"Choices: {choices}, Answer Index: {answer_idx}")
        prompts = generate_prompt(
            labels,
            self.class_names,
            n_findings_range=(1, 2),
            no_finding_name="No Finding",
        )
        
        tokens = self.tokenizer( # (4, seq_len)
            choices,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.data.text.word_num
        )
        prompt_tokens = self.tokenizer( # (seq_len,)
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.data.text.word_num
        )
        cap_len = torch.tensor(
            [int((ids != 0).sum()) for ids in prompt_tokens["input_ids"]],
            dtype=torch.long
        )

        return {
            "imgs": img,
            "caption_ids" : prompt_tokens["input_ids"][0],
            "attention_mask" : prompt_tokens["attention_mask"][0],
            "token_type_ids" : prompt_tokens["token_type_ids"][0],
            "caption_ids_all" : tokens["input_ids"],
            "attention_mask_all" : tokens["attention_mask"],
            "token_type_ids_all" : tokens["token_type_ids"],
            "cap_len_all" : cap_len,
            "label" : torch.tensor(labels, dtype=torch.float),
            "answer_idx": torch.tensor(answer_idx, dtype=torch.long),
        }

def generate_mcq_multilabel(
    labels: List[int],
    class_names: List[str],
    shuffle: bool = True
) -> Tuple[List[str], List[int], int]:
    """
    Build exactly three sentences:
        POS : "There is X."
        NEG : "There is no Y."
        HYB : "There is P but no Q."
    * A binary target array [t_pos, t_neg, t_hyb] is sampled **first** so that
      sum(targets) ≥ 1 **and** each '1' is feasible w.r.t labels.
    * Sentences are then composed to satisfy that target.
    * One of the true sentences is selected randomly as `ans_idx`.

    Returns
    -------
    sentences : list[str]  (length 3)
    targets   : list[int]  (length 3)
    ans_idx   : int
    """
    NF = len(class_names) - 1
    pos_idx = [i for i, y in enumerate(labels) if y == 1]
    neg_idx = [i for i, y in enumerate(labels) if y == 0]

    # helper lists that exclude 'No Finding' (index NF)
    pos_disease_idx = [i for i in pos_idx if i != NF]
    neg_disease_idx = [i for i in neg_idx if i != NF]

    # ------------ 0‑. No‑Finding 특수 ----------------------------
    if labels[NF] == 1 or not pos_idx:
        # allowed true patterns under NF: POS, NEG, or POS+NEG (110)
        pattern = _R.choice([(0,1,0), (1,1,0)])
        targets = list(pattern)
        
        # POS sentence
        if targets[0]:
            pos_sent = "There is No Finding."
        else:
            # false POS: random absent disease (not NF)
            pos_sent = _pos_sent(class_names[_R.choice(neg_disease_idx or range(NF))])

        # NEG sentence: random absent disease (not NF)
        neg_sent = _neg_sent(class_names[_R.choice(neg_disease_idx or range(NF))])

        # HYB sentence always false in NF: both classes from 0..NF-1
        hyb_sent = _hyb_sent(
            class_names[_R.choice(range(NF))],
            class_names[_R.choice(range(NF))]
        )
    # ------------ 1. 일반 케이스 --------------------------------
    else:
        # 1‑1. 어떤 문장들이 True 가 될 수 있는지 계산
        feasible_true = []
        if pos_idx:                    # POS true 가능
            feasible_true.append(0)
        if neg_idx:                    # NEG true 가능
            feasible_true.append(1)
        if pos_idx and neg_idx:        # HYB true 가능
            feasible_true.append(2)

        # 1‑2. 최소 1개 이상 True 되는 target 샘플
        n_true = _R.randint(1, len(feasible_true))
        true_indices = _R.sample(feasible_true, n_true)
        targets = [1 if i in true_indices else 0 for i in range(3)]

        # 1‑3. POS sentence (can be 'No Finding')
        if targets[0]:   # need a true POS → choose present class
            cls_pos = _R.choice(pos_idx)
        else:            # false POS → choose absent class
            cls_pos = _R.choice(neg_idx)
        pos_sent = _pos_sent(class_names[cls_pos])

        # 1‑4. NEG sentence (never use NF for NEG/HYB)
        if targets[1]:   # true NEG → choose absent disease (not NF)
            cls_neg = _R.choice(neg_disease_idx)
        else:            # false NEG → choose present disease (not NF)
            cls_neg = _R.choice(pos_disease_idx)
        neg_sent = _neg_sent(class_names[cls_neg])

        # 1‑5. HYB sentence (never use NF for disease slots)
        if targets[2]:   # true HYB → (present, absent) (not NF)
            cls_hyb_p = _R.choice(pos_disease_idx)
            cls_hyb_n = _R.choice(neg_disease_idx)
        else:            # false HYB → (absent, present) (not NF)
            cls_hyb_p = _R.choice(neg_disease_idx)
            cls_hyb_n = _R.choice(pos_disease_idx)
        hyb_sent = _hyb_sent(class_names[cls_hyb_p], class_names[cls_hyb_n])

    sentences = [pos_sent, neg_sent, hyb_sent]

    # 2. shuffle sentences 함께 targets 일관성 유지
    if shuffle:
        order = list(range(3))
        _R.shuffle(order)
        sentences = [sentences[i] for i in order]
        targets   = [targets[i]   for i in order]

    # 3. 정답 인덱스 : targets==1 중 무작위
    true_idxs = [i for i, t in enumerate(targets) if t == 1]
    ans_idx = _R.choice(true_idxs)

    return sentences, targets, ans_idx

# def augmentation_text_batch_random(
#     batch_labels: List[List[int]],
#     class_names: List[str],
# ) -> Tuple[List[str], np.ndarray]:
#     """
#     • 배치 내 각 이미지에 대해 참(True) 프롬프트 1개 생성
#     • 프롬프트마다 **최대 3개** 레이블만 사용 (NF 포함 가능)
#     • truth_mat[i, j] = 1  ⇒  이미지 i 에 프롬프트 j 가 참
#     """
    
#     NF = len(class_names) - 1
#     labels_np = np.asarray(batch_labels, dtype=int)
#     B, C = labels_np.shape

#     label_sets: List[Set[int]] = [
#         set(np.where(labels_np[b])[0]) - {NF} for b in range(B)
#     ]
#     batch_pool: Set[int] = set().union(*label_sets)

#     def _join(words):
#         if len(words) == 1:  return words[0]
#         if len(words) == 2:  return f"{words[0]} and {words[1]}"
#         return f"{words[0]} and {words[1]} and {words[2]}"

#     def _pos(ids): return _join([class_names[i].replace('_', ' ') for i in ids])
#     def _neg(ids): return _join([f"no {class_names[i].replace('_', ' ')}" for i in ids])

#     prompts, meta = [], []          # meta: {'pos': set, 'neg': set}
#     for L in label_sets:
#         pos_pool, neg_pool = list(L), list(batch_pool - L)

#         if not pos_pool:  # ---------- No Finding 이미지 ----------
#             mode = _R.choice(["NF", "NEG"])
#             if mode == "NF":
#                 prompts.append("There is No Finding.")
#                 meta.append({'pos': set(), 'neg': set()})
#             else:         # 최대 3개 질환 전부정문도 True
#                 k = min(3, len(neg_pool)) or 1
#                 nsel = set(_R.sample(neg_pool, k))
#                 prompts.append(f"There is {_neg(nsel)}.")
#                 meta.append({'pos': set(), 'neg': nsel})
#         else:  # ---------- 질환 존재 이미지 ---------- ## "There is Abnormal Finding"
#             mode = _R.choice(["POS", "NEG", "HYB"])
#             if mode == "POS":
#                 k = _R.randint(1, min(3, len(pos_pool)))
#                 psel = set(_R.sample(pos_pool, k))
#                 prompts.append(f"There is {_pos(psel)}.")
#                 meta.append({'pos': psel, 'neg': set()})
#             elif mode == "NEG":
#                 k = _R.randint(1, min(3, len(neg_pool))) or 1
#                 nsel = set(_R.sample(neg_pool, k))
#                 prompts.append(f"There is {_neg(nsel)}.")
#                 meta.append({'pos': set(), 'neg': nsel})
#             else:  # HYB
#                 k_pos = _R.randint(1, min(2, len(pos_pool)))
#                 k_neg = min(3 - k_pos, len(neg_pool))
#                 if k_neg == 0: k_neg = 1                  # 항상 neg 최소 1
#                 psel = set(_R.sample(pos_pool, k_pos))
#                 nsel = set(_R.sample(neg_pool, k_neg))
#                 prompts.append(f"There is {_pos(psel)} but {_neg(nsel)}.")
#                 meta.append({'pos': psel, 'neg': nsel})

#     # ---------- 참/거짓 판정 행렬 ----------
#     #  룰:
#     #    • NF 프롬프트 (m['pos']==∅ and m['neg']==∅)  →  True  ⇔  해당 이미지에 질환이 전혀 없음
#     #    • 그 외 프롬프트                      　　     →  True  ⇔  (pos ⊆ labels) ∧ (neg ∩ labels = ∅)
#     #
#     truth = np.zeros((B, len(prompts)), dtype=int)
#     for i, Li in enumerate(label_sets):
#         for j, m in enumerate(meta):
#             is_nf_prompt = (not m['pos']) and (not m['neg'])
#             if is_nf_prompt:
#                 truth[i, j] = int(len(Li) == 0)           # 이미지에 질환이 하나도 없을 때만 True
#             else:
#                 cond_pos = m['pos'].issubset(Li)          # 모든 양성 조건이 실제로 존재
#                 cond_neg = Li.isdisjoint(m['neg'])        # 음성 조건과 실제 질환이 겹치지 않음
#                 truth[i, j] = int(cond_pos and cond_neg)

#     return prompts, truth

class NIHPromptDataset(Dataset):
    def __init__(self, root, cfg, transform, df="Data_Entry_2017.csv"):
        self.root = root
        self.cfg = cfg
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)

        self.df = pd.read_csv(os.path.join(root, df))
        self.df['path'] = self.df['Image Index'].map(self._build_path_map())

        self.class_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding']
        self.prompts = {cls: f"There is {cls.replace('_', ' ')}." for cls in self.class_names}

    def _build_path_map(self):
        paths = glob(os.path.join(self.root, 'images*', '*', '*.png'))
        return {os.path.basename(p): p for p in paths}

    def __len__(self):
        return len(self.df)

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(row['path']), 0)
        img = self._resize_img(img, self.cfg.data.image.imsize)
        img = Image.fromarray(img).convert("RGB")
        img = self.transform(img)

        label_str = row['Finding Labels'].split('|')
        if 'No Finding' in label_str:
            cls_list = ['No Finding']
        else:
            cls_list = [l for l in label_str if l in self.class_names]
        
        prompt = " ".join([f"There is {cls.replace('_', ' ')}." for cls in cls_list])
        
        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.data.text.word_num
        )
        cap_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return {
            "imgs": img,
            "caption_ids" : tokens["input_ids"][0],
            "attention_mask" : tokens["attention_mask"][0],
            "token_type_ids" : tokens["token_type_ids"][0],
            "cap_len" : cap_len,
            "label" : torch.tensor([1 if cls in cls_list else 0 for cls in self.class_names], dtype=torch.float),
        }
        
class NIHPairPromptDataset(Dataset):
    def __init__(self, root, cfg, transform):
        self.root = root
        self.cfg = cfg
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)

        self.df = pd.read_csv(os.path.join(root, 'Data_Entry_2017.csv'))
        self.df['path'] = self.df['Image Index'].map(self._build_path_map())

        self.class_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding']
        self.prompts = {cls: f"There is {cls.replace('_', ' ')}." for cls in self.class_names}

    def _build_path_map(self):
        paths = glob(os.path.join(self.root, 'images*', '*', '*.png'))
        return {os.path.basename(p): p for p in paths}

    def __len__(self):
        return len(self.df)

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(row['path']), 0)
        img = self._resize_img(img, self.cfg.data.image.imsize)
        img = Image.fromarray(img).convert("RGB")
        img = self.transform(img)

        label_str = row['Finding Labels'].split('|')
        labels = torch.tensor([1 if cls in label_str else 0 for cls in self.class_names], dtype=torch.float)
        
        prompt = [f"There is {self.class_names[i]}." if label == 1 else 
              f"There is No {self.class_names[i]}." for i, label in enumerate(labels[:-1])]
        
        if labels[-1] == 1:
            prompt.append("There is No Finding.")
        
        prompt = " ".join(prompt)
        
        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.data.text.word_num
        )
        cap_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return {
            "imgs": img,
            "caption_ids" : tokens["input_ids"][0],
            "attention_mask" : tokens["attention_mask"][0],
            "token_type_ids" : tokens["token_type_ids"][0],
            "cap_len" : cap_len,
            "label" : labels,
            "prompt" : prompt,
        }
        
class NIHMCQDataset(Dataset):
    def __init__(self, root, cfg, transform):
        self.root = root
        self.cfg = cfg
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)

        self.df = pd.read_csv(os.path.join(root, 'Data_Entry_2017.csv'))
        self.df['path'] = self.df['Image Index'].map(self._build_path_map())

        self.class_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding']
        
        self.prompts = {cls: f"There is {cls.replace('_', ' ')}." for cls in self.class_names}
        self.neg_prompts = {cls: f"There is no {cls.replace('_', ' ')}." for cls in self.class_names[:-1]}
        
    def _build_path_map(self):
        paths = glob(os.path.join(self.root, 'images*', '*', '*.png'))
        return {os.path.basename(p): p for p in paths}

    def __len__(self):
        return len(self.df)

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(row['path']), 0)
        img = self._resize_img(img, self.cfg.data.image.imsize)
        img = Image.fromarray(img).convert("RGB")
        img = self.transform(img)

        label_str = row['Finding Labels'].split('|')
        labels = [1 if cls in label_str else 0 for cls in self.class_names]
        
        choices, answer_idx = generate_mcq(labels, self.class_names)
        
        tokens = self.tokenizer( # (4, seq_len)
            choices,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.data.text.word_num
        )
        cap_len = torch.tensor(
            [int((ids != 0).sum()) for ids in tokens["input_ids"]],
            dtype=torch.long
        )

        return {
            "imgs": img,
            "caption_ids_all" : tokens["input_ids"],
            "attention_mask_all" : tokens["attention_mask"],
            "token_type_ids_all" : tokens["token_type_ids"],
            "cap_len_all" : cap_len,
            "label" : torch.tensor(labels, dtype=torch.float),
            "answer_idx": torch.tensor(answer_idx, dtype=torch.long),
        }
        
class NIHMCQDataset(Dataset):
    def __init__(self, root, cfg, transform):
        self.root = root
        self.cfg = cfg
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)

        self.df = pd.read_csv(os.path.join(root, 'Data_Entry_2017.csv'))
        self.df['path'] = self.df['Image Index'].map(self._build_path_map())

        self.class_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding']
        
        self.prompts = {cls: f"There is {cls.replace('_', ' ')}." for cls in self.class_names}
        self.neg_prompts = {cls: f"There is no {cls.replace('_', ' ')}." for cls in self.class_names[:-1]}
        
    def _build_path_map(self):
        paths = glob(os.path.join(self.root, 'images*', '*', '*.png'))
        return {os.path.basename(p): p for p in paths}

    def __len__(self):
        return len(self.df)

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(row['path']), 0)
        img = self._resize_img(img, self.cfg.data.image.imsize)
        img = Image.fromarray(img).convert("RGB")
        img = self.transform(img)

        label_str = row['Finding Labels'].split('|')
        labels = [1 if cls in label_str else 0 for cls in self.class_names]
        
        choices, answer_idx = generate_mcq(labels, self.class_names)
        
        tokens = self.tokenizer( # (4, seq_len)
            choices,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.data.text.word_num
        )
        cap_len = torch.tensor(
            [int((ids != 0).sum()) for ids in tokens["input_ids"]],
            dtype=torch.long
        )

        return {
            "imgs": img,
            "caption_ids_all" : tokens["input_ids"],
            "attention_mask_all" : tokens["attention_mask"],
            "token_type_ids_all" : tokens["token_type_ids"],
            "cap_len_all" : cap_len,
            "label" : torch.tensor(labels, dtype=torch.float),
            "answer_idx": torch.tensor(answer_idx, dtype=torch.long),
        }

class NIHMultiLabelMCQDataset(Dataset):
    def __init__(self, root, cfg, transform):
        self.root = root
        self.cfg = cfg
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)

        self.df = pd.read_csv(os.path.join(root, 'Data_Entry_2017.csv'))
        self.df['path'] = self.df['Image Index'].map(self._build_path_map())

        self.class_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding']
        
        self.prompts = {cls: f"There is {cls.replace('_', ' ')}." for cls in self.class_names}
        
    def _build_path_map(self):
        paths = glob(os.path.join(self.root, 'images*', '*', '*.png'))
        return {os.path.basename(p): p for p in paths}

    def __len__(self):
        return len(self.df)

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(row['path']), 0)
        img = self._resize_img(img, self.cfg.data.image.imsize)
        img = Image.fromarray(img).convert("RGB")
        img = self.transform(img)

        label_str = row['Finding Labels'].split('|')
        labels = [1 if cls in label_str else 0 for cls in self.class_names]
        
        choices, targets, answer_idx = generate_mcq_multilabel(labels, self.class_names)
        
        tokens = self.tokenizer( # (4, seq_len)
            choices,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.data.text.word_num
        )
        cap_len = torch.tensor(
            [int((ids != 0).sum()) for ids in tokens["input_ids"]],
            dtype=torch.long
        )

        return {
            "imgs": img,
            "caption_ids_all" : tokens["input_ids"],
            "attention_mask_all" : tokens["attention_mask"],
            "token_type_ids_all" : tokens["token_type_ids"],
            "cap_len_all" : cap_len,
            "label" : torch.tensor(labels, dtype=torch.float),
            "answer_idx": torch.tensor(answer_idx, dtype=torch.long),
            "targets" : torch.tensor(targets, dtype=torch.float)
        }
        
class NIHConCARZeroDataset(Dataset):
    def __init__(self, root, cfg, transform):
        self.root = root
        self.cfg = cfg
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)

        self.df = pd.read_csv(os.path.join(root, 'Data_Entry_2017.csv'))
        self.df['path'] = self.df['Image Index'].map(self._build_path_map())

        self.class_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding']
        
        self.prompts = {cls: f"There is {cls.replace('_', ' ')}." for cls in self.class_names}
        
    def _build_path_map(self):
        paths = glob(os.path.join(self.root, 'images*', '*', '*.png'))
        return {os.path.basename(p): p for p in paths}

    def __len__(self):
        return len(self.df)

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )
        return resized_img
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(row['path']), 0)
        img = self._resize_img(img, self.cfg.data.image.imsize)
        img = Image.fromarray(img).convert("RGB")
        img = self.transform(img)

        label_str = row['Finding Labels'].split('|')
        labels = [1 if cls in label_str else 0 for cls in self.class_names]
        
        choices, targets, answer_idx = generate_mcq_multilabel(labels, self.class_names)
        
        tokens = self.tokenizer( # (4, seq_len)
            choices,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.data.text.word_num
        )
        
        cap_len = torch.tensor(
            [int((ids != 0).sum()) for ids in tokens["input_ids"]],
            dtype=torch.long
        )

        return {
            "imgs": img,
            "caption_ids_all" : tokens["input_ids"],
            "attention_mask_all" : tokens["attention_mask"],
            "token_type_ids_all" : tokens["token_type_ids"],
            "cap_len_all" : cap_len,
            "label" : torch.tensor(labels, dtype=torch.float),
            "answer_idx": torch.tensor(answer_idx, dtype=torch.long),
            "targets" : torch.tensor(targets, dtype=torch.float)
        }
        
class NIHDataset(Dataset):
    def __init__(self, df, cfg, transform):
        self.df = df
        self.cfg = cfg
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )
        return resized_img
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(row['Path']), 0)
        img = self._resize_img(img, self.cfg.data.image.imsize)
        img = Image.fromarray(img).convert("RGB")
        img = self.transform(img)

        labels = row.iloc[2:].tolist()

        return {
            "imgs": img,
            "label" : torch.tensor(labels, dtype=torch.float),
        }

def augmentation_text_batch_random(
    batch_labels: List[List[int]],
    class_names: List[str],
    prob_cfg: Optional[Dict[str, Sequence[float]]] = None,
) -> Tuple[List[str], np.ndarray]:
    """
    • 배치 내 각 이미지에 대해 참(True) 프롬프트 1개 생성
    • 프롬프트마다 최대 3개 레이블 사용 (NF 포함 가능)
    • truth_mat[i, j] = 1  ⇒  이미지 i 에 프롬프트 j 가 참
    """

    # ───── probability config (epoch‑dependent) ─────
    if prob_cfg is None:
        prob_cfg = {}
    w_neg_nf       = prob_cfg.get("neg_nf",       [0.5, 0.5])         # ["NF","NEG"]
    w_pos_neg_hyb  = prob_cfg.get("pos_neg_hyb",  [1/3, 1/3, 1/3])    # ["POS","NEG","HYB"]
    w_sub_spec_abn = prob_cfg.get("sub_spec_abn", [0.8, 0.2])         # ["SPEC","ABN"]

    NF  = len(class_names) - 1
    ABN = -2                                 # ☑️ Sentinel 라벨 (절대 실제 라벨과 겹치지 않음)

    labels_np = np.asarray(batch_labels, dtype=int)
    B, C      = labels_np.shape

    label_sets: List[Set[int]] = [
        set(np.where(labels_np[b])[0]) - {NF} for b in range(B)
    ]
    batch_pool: Set[int] = set().union(*label_sets)

    def _join(words):
        if len(words) == 1: return words[0]
        if len(words) == 2: return f"{words[0]} and {words[1]}"
        return f"{words[0]} and {words[1]} and {words[2]}"

    _pos = lambda ids: _join([class_names[i].replace('_', ' ') for i in ids])
    _neg = lambda ids: _join([f"no {class_names[i].replace('_', ' ')}" for i in ids])

    prompts, meta = [], []                   # meta: {'pos': set, 'neg': set}
    for L in label_sets:
        pos_pool, neg_pool = list(L), list(batch_pool - L)

        # ---------- No Finding 이미지 ----------
        if not pos_pool:
            mode = _R.choices(["NF", "NEG"], weights=w_neg_nf, k=1)[0]
            if mode == "NF":
                prompts.append("There is No Finding.")
                meta.append({'pos': set(), 'neg': set()})
            else:
                k     = min(3, len(neg_pool)) or 1
                nsel  = set(_R.sample(neg_pool, k))
                prompts.append(f"There is {_neg(nsel)}.")
                meta.append({'pos': set(), 'neg': nsel})

        # ---------- 질환 존재 이미지 ----------
        else:
            if len(neg_pool) == 0:
                mode = _R.choices(["POS"], weights=[1.0], k=1)[0]
            else:
                mode = _R.choices(["POS", "NEG", "HYB"], weights=w_pos_neg_hyb, k=1)[0]
            if mode == "POS":
                # ☑️ POS 내부 두 가지 하위 모드 중 랜덤 선택
                sub = _R.choices(["SPEC", "ABN"], weights=w_sub_spec_abn, k=1)[0]
                if sub == "ABN":             # ---- Abnormal Finding 모드
                    prompts.append("There is Abnormal Finding.")
                    meta.append({'pos': {ABN}, 'neg': set()})
                else:                        # ---- 기존 SPEC 모드
                    k     = _R.randint(1, min(3, len(pos_pool)))
                    psel  = set(_R.sample(pos_pool, k))
                    prompts.append(f"There is {_pos(psel)}.")
                    meta.append({'pos': psel, 'neg': set()})

            elif mode == "NEG":
                k     = _R.randint(1, min(3, len(neg_pool)))
                nsel  = set(_R.sample(neg_pool, k))
                prompts.append(f"There is {_neg(nsel)}.")
                meta.append({'pos': set(), 'neg': nsel})

            else:  # HYB
                k_pos = _R.randint(1, min(2, len(pos_pool)))
                k_neg = min(3 - k_pos, len(neg_pool))
                if k_neg == 0: k_neg = 1
                psel  = set(_R.sample(pos_pool, k_pos))
                nsel  = set(_R.sample(neg_pool, k_neg))
                prompts.append(f"There is {_pos(psel)} but {_neg(nsel)}.")
                meta.append({'pos': psel, 'neg': nsel})

    # ---------- 참/거짓 판정 행렬 ----------
    truth = np.zeros((B, len(prompts)), dtype=int)
    for i, Li in enumerate(label_sets):
        for j, m in enumerate(meta):
            if ABN in m['pos']:
                truth[i, j] = int(len(Li) > 0)
                continue

            is_nf_prompt = (not m['pos']) and (not m['neg'])
            if is_nf_prompt:
                truth[i, j] = int(len(Li) == 0)
            else:
                cond_pos = m['pos'].issubset(Li)
                cond_neg = Li.isdisjoint(m['neg'])
                truth[i, j] = int(cond_pos and cond_neg)

    return prompts, truth
        
class PromptBatchCollator:
    def __init__(self, class_names, tokenizer, seq_len,
                 prob_sched: Optional[Callable[[int], Dict[str, Sequence[float]]]] = None):
        self.class_names = class_names
        self.tokenizer   = tokenizer
        self.seq_len     = seq_len
        self.prob_sched = prob_sched or (lambda epoch: {})
        self.cur_epoch  = 0
        self.prob_cfg = self.prob_sched(self.cur_epoch)  # 초기화 시점에 prob_cfg 설정

    def set_epoch(self, epoch: int):
        """Must be called at every epoch start by the training loop."""
        self.cur_epoch = epoch
        self.prob_cfg = self.prob_sched(self.cur_epoch)

    def __call__(self, batch):
        """
        batch: List[ dict(img, label, …) ]  (dataset 에서 반환된 항목)
        반환:   dict(imgs, labels, prompt_ids, prompt_mask, truth_mat)
        """
        imgs   = [b['imgs']  for b in batch]           # (B,3,H,W)
        labels = [b['label'].tolist() for b in batch]  # List[List[int]] length = C

        # ── ① 프롬프트·truth 생성 (배치 단위) ─────────────────────────
        prompts, truth = augmentation_text_batch_random(labels, self.class_names, self.prob_cfg)
        # prompts: List[str] (N) / truth: ndarray (B,N)

        tok = self.tokenizer(
            prompts,
            padding='max_length',
            truncation=True,
            max_length=self.seq_len,
            return_tensors='pt'
        )                                             # (N,L)

        out = default_collate(batch)                  # 기타 기본 키 결합
        out.update({
            "imgs"        : torch.stack(imgs),        # (B,3,H,W)
            "truth"       : torch.tensor(truth),      # (B,N)
            "caption_ids"  : tok["input_ids"],         # (N,L)
            "attention_mask" : tok["attention_mask"],    # (N,L)
            "cap_len" : torch.tensor([b['cap_len'] for b in batch], dtype=torch.int32),  # (B,)
            "token_type_ids" : tok["token_type_ids"],    # (N,L)
        })
        return out

# ────────────────────────────────────────────────
# Example: linear decay/boost of probabilities
# ────────────────────────────────────────────────
def linear_prob_scheduler(max_epoch: int):
    """
    Returns a scheduler: epoch ↦ prob_cfg dict.
    NF prompts become more frequent, NEG less frequent, etc.
    """
    def sched(epoch: int):
        t = min(max(epoch / max_epoch, 0.0), 1.0)
        return {
            "neg_nf": [1.0, 0.0],                       # ["NF","NEG"]
            "pos_neg_hyb": [1/3, 1/3, 1/3],
            #"sub_spec_abn": [0.5+t/2, 0.5-t/2]  # ["SPEC","ABN"]
            "sub_spec_abn": [1.0, 0.0]  # ["SPEC","ABN"]
        }
    return sched

def _assign_representative_labels(labels, class_names):
    """
    coverage-first + balance-second:
    1) 배치에 등장한 각 질환이 최소 1개 샘플에 할당되도록 보장
    2) 남은 샘플은 현재까지 할당 횟수가 가장 적은 질환을 선택
    반환: rep_lbl_idx (길이 B, 각 이미지의 대표 질환 인덱스 또는 None(No Finding 전용))
    """
    B, C = labels.shape
    NF   = class_names.index('No Finding')

    # 각 이미지의 양성 질환 인덱스 셋(정상은 공집합)
    pos_sets = [set([i for i, v in enumerate(lbl.tolist()) if v == 1 and i != NF])
                for lbl in labels]

    # 배치 내 등장 질환(정상 제외)
    present_classes = sorted({c for s in pos_sets for c in s})

    rep = [None] * B          # 대표 질환 인덱스
    used_count = {c: 0 for c in present_classes}

    # 1) coverage-first: 희귀 질환부터 배정
    # (희귀 = 등장 이미지 수가 적은 질환)
    rarity = {c: sum([1 for s in pos_sets if c in s]) for c in present_classes}
    for c in sorted(present_classes, key=lambda x: rarity[x]):
        # 아직 대표를 정하지 않은 이미지 중 이 질환을 가진 샘플 찾기
        cand = [i for i, s in enumerate(pos_sets) if (rep[i] is None) and (c in s)]
        if cand:
            i = random.choice(cand)
            rep[i] = c
            used_count[c] += 1

    # 2) balance-second: 나머지 샘플은 현재 사용 빈도가 가장 낮은 질환으로
    for i, s in enumerate(pos_sets):
        if rep[i] is None and len(s) > 0:
            # 사용 빈도가 가장 낮은 질환 선택 (tie 시 랜덤)
            min_used = min(used_count.get(c, 0) for c in s)
            cand = [c for c in s if used_count.get(c, 0) == min_used]
            c = random.choice(cand)
            rep[i] = c
            used_count[c] = used_count.get(c, 0) + 1

    # 정상만 있는 샘플은 None 유지
    return rep, present_classes

def _parse_prompt(p: str):
    assert p.startswith("There is ")
    body = p[len("There is "):].rstrip(".")
    if body == "No Finding":
        return ("No Finding", True)
    if body.startswith("no "):
        return (body[3:].replace(" ", "_"), False)   # 'There is no A.'
    return (body.replace(" ", "_"), True)            # 'There is A.'

def _build_truth_ip_from_prompts(prompts, labels, class_names):
    name2idx = {n: i for i, n in enumerate(class_names)}
    NF = name2idx["No Finding"]
    B = labels.size(0); P = len(prompts)
    truth_ip = torch.zeros(B, P, dtype=torch.bool)
    lbl = labels > 0.5
    for p_idx, p in enumerate(prompts):
        cls, exists = _parse_prompt(p)
        if cls == "No_Finding":
            mask = (lbl[:, NF] & (lbl.sum(dim=1) == 1))
        else:
            k = name2idx[cls]
            mask = lbl[:, k] if exists else (~lbl[:, k])
        truth_ip[:, p_idx] = mask
    return truth_ip

def _prompt_to_class_idx(p: str, class_names):
    assert p.startswith("There is ")
    body = p[len("There is "):].rstrip(".")
    # 부정문이면 'no ' 제거 (타깃 클래스명만 남김)
    if body.startswith("no "):
        body = body[3:]
    # 'No Finding'은 그대로, 나머지는 공백→언더스코어로 되돌림
    if body == "No Finding":
        key = "No Finding"
    else:
        key = body.replace(" ", "_")
    try:
        return class_names.index(key)
    except ValueError:
        raise ValueError(f"[prompt->class] '{p}'에서 얻은 클래스 '{key}'가 class_names에 없습니다.")

def nih_collate_with_truth(batch, tokenizer, class_names):
    imgs   = torch.stack([b["imgs"]  for b in batch])      # [B,3,H,W]
    labels = torch.stack([b["label"] for b in batch])      # [B,15]
    B, C   = labels.shape
    NF     = class_names.index('No Finding')

    # 1) 대표 질환 할당 (비정상은 단일 대표, 정상은 None)
    rep, present_classes = _assign_representative_labels(labels, class_names)

    # 배치 내 존재 질환(정상 제외)
    present_non_nf = [c for c in present_classes if c != NF]

    # 이미 "양성 프롬프트"로 사용된 질환 집합(정상 제외)
    used_pos_non_nf = {r for r in rep if r is not None}

    # NF 부정 프롬프트 우선 후보: (배치 존재) ∩ (양성 프롬프트 미사용)
    nf_neg_pool = list(set(present_non_nf) - used_pos_non_nf)

    pos_prompts, neg_prompts = [], []
    pos_types,   neg_types   = [], []   # 'pos' / 'neg' 표기용

    for i in range(B):
        lbl = labels[i]
        is_nf = bool((lbl[NF] > 0.5) and (lbl.sum() == 1))

        # -------- 양성 프롬프트 --------
        if is_nf:
            pos_sentence = "There is No Finding."
        else:
            c_idx = rep[i]; assert c_idx is not None
            pos_sentence = f"There is {class_names[c_idx].replace('_',' ')}."
        pos_prompts.append(pos_sentence)
        pos_types.append('pos')

        # -------- 부정/반대 프롬프트 처리 --------
        if is_nf:
            # 기존: NF의 "부정 프롬프트"로 질환 긍정문을 넣었으나,
            # 요구사항: 그 질환 긍정문을 'pos'로 추가하고, 그에 대응하는 부정문을 'neg'로 추가.
            # 1순위: (배치 존재) ∩ (양성 프롬프트 미사용)
            if len(nf_neg_pool) > 0:
                pick = random.randrange(len(nf_neg_pool))
                c_idx_neg = nf_neg_pool.pop(pick)
            # 2순위: 배치 존재 질환
            elif len(present_non_nf) > 0:
                c_idx_neg = random.choice(present_non_nf)
            # 3순위: 전체(정상 제외) 랜덤
            else:
                c_idx_neg = random.randint(0, C-2)

            # (A) 질환 긍정문을 'pos'로 추가
            extra_pos = f"There is {class_names[c_idx_neg].replace('_',' ')}."
            pos_prompts.append(extra_pos)
            pos_types.append('pos')

            # (B) 해당 질환의 부정문을 'neg'로 추가
            extra_neg = f"There is no {class_names[c_idx_neg].replace('_',' ')}."
            neg_prompts.append(extra_neg)
            neg_types.append('neg')

        else:
            # 비정상: 대표 질환 부정문만 추가
            neg_sentence = f"There is no {class_names[rep[i]].replace('_',' ')}."
            neg_prompts.append(neg_sentence)
            neg_types.append('neg')

    # 2) 프롬프트 결합 (NF 케이스로 인해 길이는 2B 이상이 될 수 있음)
    all_prompts = pos_prompts + neg_prompts
    all_types   = pos_types   + neg_types

    # 3) 중복 제거 (stable; 최초 등장 유지, pos/neg 혼재 시 'both')
    seen = {}
    uniq_prompts, uniq_types = [], []
    orig2uniq = []   # 원본 인덱스(길이=len(all_prompts)) → 유니크 열 인덱스
    for j, p in enumerate(all_prompts):
        if p not in seen:
            seen[p] = len(uniq_prompts)
            uniq_prompts.append(p)
            uniq_types.append(all_types[j])     # 'pos' 또는 'neg'로 기록
        else:
            u = seen[p]
            if uniq_types[u] != all_types[j]:
                uniq_types[u] = 'both'          # 양·음성 양쪽에서 등장
        orig2uniq.append(seen[p])

    # 4) 토크나이즈 (유니크 프롬프트 기준)
    tok = tokenizer(
        uniq_prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max(64, max(b["cap_len"] for b in batch))
    )  # [P_u, L]

    # 5) truth 생성: 유니크 프롬프트 기준 [B, P_u]
    truth_ip = _build_truth_ip_from_prompts(uniq_prompts, labels, class_names)
    
    prompt_target_idx  = [ _prompt_to_class_idx(p, class_names) for p in uniq_prompts ]

    return {
        "imgs"           : imgs,                         # [B,3,H,W]
        "labels"         : labels.float(),               # [B,15]
        "caption_ids"    : tok["input_ids"],             # [P_u,L]
        "attention_mask" : tok["attention_mask"],        # [P_u,L]
        "token_type_ids" : tok["token_type_ids"],      # [P_u,L]
        "truth"          : truth_ip,                     # [B,P_u]
        "prompts"        : uniq_prompts,                 # list[str], len=P_u
        "prompt_type"    : uniq_types,                   # list['pos'|'neg'|'both']
        "prompt_target_idx" : prompt_target_idx,         # list[int], 길이=P_u, 각 프롬프트의 타깃 클래스 인덱스
    }
    
class NIHDCLDataset(Dataset):
    """
    getitem은 '샘플-특이' 항목만 반환하고,
    프롬프트 관련 '정적 상수'는 collate_fn에서 배치에 1회 주입한다.
    반환(배치 단위):
      - imgs:                [B, C, H, W]
      - labels:              [B, 15]
      - prompt_truth:        [B, 29]
      - prompts:             dict[idx]-> {text, input_ids, attention_mask, token_type_ids(or None),
                                          length, target_idx, type}  # 배치 크기와 무관, 29개 고정
    """
    def __init__(self, root, cfg, transform):
        super().__init__()
        self.root = root
        self.cfg = cfg
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)

        # ── 메타 로드 ──────────────────────────────────────────────────────────
        self.df = pd.read_csv(os.path.join(root, 'Data_Entry_2017.csv'))
        self.df['path'] = self.df['Image Index'].map(self._build_path_map())

        # ── 클래스/프롬프트 정의 ─────────────────────────────────────────────
        self.class_names = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
            'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
            'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding'
        ]
        self.disease_names = self.class_names[:-1]  # 14개
        self.nf_idx = len(self.class_names) - 1     # 14

        pos_prompts = [f"There is {c.replace('_', ' ')}." for c in self.disease_names]
        nf_prompt   = ["There is No Finding."]
        neg_prompts = [f"There is no {c.replace('_', ' ')}." for c in self.disease_names]
        self.prompt_texts = pos_prompts + nf_prompt + neg_prompts  # 총 29
        self.prompts = {str(i): prompt for i,prompt in enumerate(pos_prompts+nf_prompt)}
        self.neg_prompts = {str(i): prompt for i,prompt in enumerate(neg_prompts)}

        self.prompt_target_idx = (
            list(range(len(self.disease_names))) + [self.nf_idx] + list(range(len(self.disease_names)))
        )
        self.prompt_type = [1]*len(self.disease_names) + [1] + [0]*len(self.disease_names)

        # ── 프롬프트 토큰을 '정적 상수'로 미리 준비 ───────────────────────────
        max_len = self.cfg.data.text.word_num
        toks = self.tokenizer(
            self.prompt_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_len
        )
        self.prompt_input_ids      = toks["input_ids"]                 # [29, L]
        self.prompt_attention_mask = toks["attention_mask"]            # [29, L]
        self.prompt_token_type_ids = toks.get("token_type_ids", None)  # [29, L] or None
        self.prompt_lengths        = self.prompt_attention_mask.sum(dim=1)  # [29]

        # (선택) 딕셔너리 형태로 한번 구성해두면, collate에서 그대로 건네줄 수 있다.
        self.prompts_static = {}
        for i in range(len(self.prompt_texts)):
            self.prompts_static[i] = {
                "text": self.prompt_texts[i],
                "input_ids":      self.prompt_input_ids[i],          # [L]
                "attention_mask": self.prompt_attention_mask[i],     # [L]
                "token_type_ids": None if self.prompt_token_type_ids is None else self.prompt_token_type_ids[i],  # [L] or None
                "length": int(self.prompt_lengths[i].item()),
                "target_idx": int(self.prompt_target_idx[i]),
                "type": int(self.prompt_type[i]),   # 1=긍정, 0=부정
            }

    def _build_path_map(self):
        paths = glob(os.path.join(self.root, 'images*', '*', '*.png'))
        # paths += glob(os.path.join(self.root, 'images*', '*', '*.jpg'))  # 필요 시 추가
        return {os.path.basename(p): p for p in paths}

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _resize_img(img, scale):
        H, W = img.shape[:2]
        if H >= W:
            sf = scale / float(H)
            newW = int(round(W * sf))
            resized = cv2.resize(img, (newW, scale), interpolation=cv2.INTER_AREA)
            pad = scale - newW
            left, right = pad // 2, pad - pad // 2
            top, bottom = 0, 0
        else:
            sf = scale / float(W)
            newH = int(round(H * sf))
            resized = cv2.resize(img, (scale, newH), interpolation=cv2.INTER_AREA)
            pad = scale - newH
            top, bottom = pad // 2, pad - pad // 2
            left, right = 0, 0
        return np.pad(resized, ((top, bottom), (left, right)), mode='constant', constant_values=0)

    def _row_to_labels(self, row):
        labs = torch.zeros(len(self.class_names), dtype=torch.float32)
        lbls = [s.strip() for s in str(row['Finding Labels']).split('|')]
        if 'No Finding' in lbls:
            labs[self.nf_idx] = 1.0
        else:
            for c in lbls:
                if c in self.disease_names:
                    labs[self.disease_names.index(c)] = 1.0
            labs[self.nf_idx] = 0.0
        return labs

    def _truth_for_prompts(self, labels_15):
        y_dis = labels_15[:len(self.disease_names)]    # [14]
        y_nf  = labels_15[self.nf_idx].item()          # scalar
        truth_pos = y_dis.clone()                      # [14]
        truth_nf  = torch.tensor([y_nf], dtype=torch.float32)
        truth_neg = (1.0 - y_dis)                      # [14]
        return torch.cat([truth_pos, truth_nf, truth_neg], dim=0)  # [29]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(row['path']), cv2.IMREAD_GRAYSCALE)
        img = self._resize_img(img, self.cfg.data.image.imsize)
        img = Image.fromarray(img).convert("RGB")
        img = self.transform(img)  # Tensor[C,H,W]

        labels = self._row_to_labels(row)          # [15]
        prompt_truth = self._truth_for_prompts(labels)  # [29]

        # ⚠️ 프롬프트 토큰 관련 텐서는 반환하지 않는다 (정적 상수로 collate에서 주입)
        return {
            "imgs": img,
            "labels": labels,
            "prompt_truth": prompt_truth,  # 샘플 고유 항목이므로 유지
        }

    # ── 핵심: 배치 결합 함수 (프롬프트 정적 상수 주입) ─────────────────────────
    def collate_fn(self, batch):
        imgs         = torch.stack([b["imgs"] for b in batch], dim=0)            # [B,C,H,W]
        labels       = torch.stack([b["labels"] for b in batch], dim=0)          # [B,15]
        prompt_truth = torch.stack([b["prompt_truth"] for b in batch], dim=0)    # [B,29]

        # '정적' 프롬프트 정보: 배치마다 1회만 포함 (배치 차원 없음)
        batch_out = {
            "imgs": imgs,
            "labels": labels,
            "prompt_truth": prompt_truth,

            # ① 딕셔너리 형태(요청사항)
            "prompts": self.prompts_static,

            # ② 필요시 텐서 형태로도 병행 제공(고정 크기, 배치 차원 없음)
            "caption_ids":      self.prompt_input_ids,       # [29, L]
            "attention_mask": self.prompt_attention_mask,  # [29, L]
            "token_type_ids": self.prompt_token_type_ids,  # [29, L] or None
            "cap_len":        self.prompt_lengths,         # [29]
            "prompt_target_idx":     torch.tensor(self.prompt_target_idx, dtype=torch.long),
            "prompt_type":           torch.tensor(self.prompt_type, dtype=torch.long),
        }
        return batch_out
    
class NIHPosNegDataset(Dataset):
    def __init__(self, df, cfg, transform):
        self.df = df
        self.cfg = cfg
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)

        self.class_names = ['Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Pulmonary Infiltration', 'Pulmonary Mass', 'Lung Nodule', 'Pneumonia', 'Pneumothorax',
            'Pulmonary Consolidation', 'Pulmonary Edema', 'Pulmonary Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia', 'no finding']
        self.pos_prompts = {cls: f"There is {cls.replace('_', ' ')}." for cls in self.class_names}
        self.neg_prompts = {cls: f"There is no {cls.replace('_', ' ')}." for cls in self.class_names[:-1]}

    def __len__(self):
        return len(self.df)

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )
        return resized_img
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(row['Path']), 0)
        img = self._resize_img(img, self.cfg.data.image.imsize)
        img = Image.fromarray(img).convert("RGB")
        img = self.transform(img)

        labels = row.iloc[2:].tolist()
        
        if self.cfg.data.text.neg :
            choices, answer_idx = generate_mcq2(labels, self.class_names, no_hyb=True)
            choices = choices[answer_idx]
        else :
            if labels[-1] == 1 :
                choices = f"There is no finding."
            else :
                pos_indices = [i for i, v in enumerate(labels[:-1]) if v == 1]
                chosen_idx = random.choice(pos_indices)
                chosen_class = self.class_names[chosen_idx]
                choices = f"There is {chosen_class.replace('_', ' ')}."
        
        tokens = self.tokenizer( # (4, seq_len)
            choices,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.data.text.word_num
        )
        
        
        cap_len = torch.tensor(
            [int((ids != 0).sum()) for ids in tokens["input_ids"]],
            dtype=torch.long
        )
        
        return {
            "imgs": img,
            "caption_ids" : tokens["input_ids"][0],
            "attention_mask" : tokens["attention_mask"][0],
            "token_type_ids" : tokens["token_type_ids"][0],
            "cap_len" : cap_len,
            "label" : torch.tensor(labels, dtype=torch.float),
        }