import torch
import torch.nn as nn
import cv2
import re
import numpy as np
from sklearn import metrics

from PIL import Image
from .. import builder
from .. import loss
from .. import utils
from transformers import AutoTokenizer
from nltk.tokenize import RegexpTokenizer
import ipdb

def build_block_mask(batch_sz: int, n_rep: int, txt_len: int, device):
    """
    이미지 i 의 모든 쿼리토큰이  ▶︎  텍스트 (i*4 ~ i*4+3) 프롬프트에만
    Attend 하도록 True 로 표시한 bool mask.

    반환 shape : (batch_sz*n_rep, 1, txt_len)
                 = MHA / SDPA 의 (q_seq, 1, k_seq) 형식
    """
    mask = torch.zeros(batch_sz * n_rep, txt_len, dtype=torch.bool, device=device)
    for i in range(batch_sz):
        col = slice(i * n_rep, (i + 1) * n_rep)   # i*4 ~ i*4+3
        mask[i * n_rep : (i+1) * n_rep, col] = True
    return mask.unsqueeze(1)   # (B*4, 1, Tk)

class CARZeroDQNWOSAGLMLP(nn.Module):
    def __init__(self, cfg):
        super(CARZeroDQNWOSAGLMLP, self).__init__()

        self.cfg = cfg
        self.text_encoder = builder.build_text_model(cfg)
        self.img_encoder = builder.build_img_model(cfg)
        self.i2t_cls_fuse_mlp = nn.Sequential(
            nn.Linear(cfg.model.text.embedding_dim*2, cfg.model.text.embedding_dim),
            nn.GELU(),
            nn.Linear(cfg.model.text.embedding_dim, cfg.model.text.embedding_dim),
        )
        self.i2t_cls_local_gating = nn.Linear(cfg.model.text.embedding_dim*2, cfg.model.text.embedding_dim)
        self.i2t_fusion_module = builder.build_dqn_wo_self_atten_mlp_module(cfg, cfg.model.fusion.i2t_average_attn_weights)
        if self.cfg.model.CARZero.shared_fusion :
            self.t2i_fusion_module = self.i2t_fusion_module
        else:
            self.t2i_fusion_module = builder.build_dqn_wo_self_atten_mlp_module(cfg, cfg.model.fusion.t2i_average_attn_weights)

        self.local_loss = loss.CARZero_loss.local_loss
        self.global_loss = loss.CARZero_loss.global_loss
        self.local_loss_weight = self.cfg.model.CARZero.local_loss_weight
        self.global_loss_weight = self.cfg.model.CARZero.global_loss_weight
        self.ce_loss_weight = self.cfg.model.CARZero.ce_loss_weight

        self.temp1 = self.cfg.model.CARZero.temp1
        self.temp2 = self.cfg.model.CARZero.temp2
        self.temp3 = self.cfg.model.CARZero.temp3
        self.batch_size = self.cfg.train.batch_size

        self.ce_loss = loss.dqn_cos_loss.DQNCOSLoss()
        self.weighted_ce_loss = loss.dqn_cos_loss.MatrixInfoNCELoss()

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)
        self.ixtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
        
        self.i2t_tau = nn.Parameter(torch.log(torch.tensor(1.0)))
        self.t2i_tau = nn.Parameter(torch.log(torch.tensor(1.0)))
        
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        embed_dim = cfg.model.fusion.d_model
        class_num = cfg.model.fusion.class_num
        
        self.edl_head = nn.Sequential(
            nn.Linear(2*embed_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, class_num)
        )

    def text_encoder_forward(self, caption_ids, attention_mask, token_type_ids):
        text_emb_l, text_emb_g, sents = self.text_encoder(
            caption_ids, attention_mask, token_type_ids)
        return text_emb_l, text_emb_g, sents

    def image_encoder_forward(self, imgs):
        img_feat_g, img_emb_l = self.img_encoder(imgs, get_local=True)
        img_emb_g, img_emb_l = self.img_encoder.generate_embeddings(
            img_feat_g, img_emb_l
        )
        return img_emb_l, img_emb_g

    def _calc_local_loss(self, img_emb_l, text_emb_l, sents):
        
        # ipdb.set_trace()
        cap_lens = [
            len([w for w in sent if not w.startswith("[")]) + 1 for sent in sents
        ]
        l_loss0, l_loss1, attn_maps = self.local_loss(
            img_emb_l,
            text_emb_l,
            cap_lens,
            temp1=self.temp1,
            temp2=self.temp2,
            temp3=self.temp3,
        )
        return l_loss0, l_loss1, attn_maps

    def _calc_global_loss(self, img_emb_g, text_emb_g):
        g_loss0, g_loss1 = self.global_loss(img_emb_g, text_emb_g, temp3=self.temp3)
        return g_loss0, g_loss1

    def _calc_ce_loss(self, cls):
        loss = self.ce_loss(cls)
        return loss

    def _calc_weighted_ce_loss(self, cls, weight):
        loss = self.weighted_ce_loss(cls, weight)
        return loss
    
    def calc_loss(self, i2t_cls, t2i_cls):
        # l_loss0, l_loss1, attn_maps = self._calc_local_loss(
        #     img_emb_l, text_emb_l, sents
        # )
        # g_loss0, g_loss1 = self._calc_global_loss(img_emb_g, text_emb_g)

        # # weighted loss
        loss = 0
        # loss += (l_loss0 + l_loss1) * self.local_loss_weight
        # loss += (g_loss0 + g_loss1) * self.global_loss_weight

        ce_loss0 = self._calc_ce_loss(i2t_cls)
        ce_loss1 = self._calc_ce_loss(t2i_cls)
        loss += ce_loss0 * self.ce_loss_weight
        loss += ce_loss1 * self.ce_loss_weight

        return loss
    
    def calc_loss_aug(self, i2t_cls, t2i_cls, weight):

        # l_loss0, l_loss1, attn_maps = self._calc_local_loss(
        #     img_emb_l, text_emb_l, sents
        # )
        # g_loss0, g_loss1 = self._calc_global_loss(img_emb_g, text_emb_g)

        # # weighted loss
        loss = 0
        # loss += (l_loss0 + l_loss1) * self.local_loss_weight
        # loss += (g_loss0 + g_loss1) * self.global_loss_weight

        ce_loss0 = self._calc_weighted_ce_loss(i2t_cls, weight)
        ce_loss1 = self._calc_weighted_ce_loss(t2i_cls, weight)
        loss += ce_loss0 * self.ce_loss_weight
        loss += ce_loss1 * self.ce_loss_weight

        return loss

    def forward(self, x, feat=False):
        # img encoder branch
        img_emb_l, img_emb_g = self.image_encoder_forward(x["imgs"])

        # text encorder branch
        text_emb_l, text_emb_g, sents = self.text_encoder_forward(
            x["caption_ids"], x["attention_mask"], x["token_type_ids"]
        )
        
        # if self.cfg.mcq : 
        #     B = x["imgs"].size(0) // 4
        #     N = 4
        #     Tk = text_emb_g.size(0)
        #     attn_mask = build_block_mask(B, N, Tk, device=text_emb_g.device)
            
        #     img_emb_l_ = img_emb_l.flatten(2).permute(0, 2, 1)        # (B*4, patch, D)
        #     txt_emb_l_ = text_emb_l.flatten(2).permute(0, 2, 1)            # (B*4, token, D)

        #     i2t_cls = self.fusion_module(                    # ▸ Query = 이미지
        #         torch.cat([img_emb_g.unsqueeze(1), img_emb_l_], dim=1),
        #         text_emb_g,
        #         attn_mask = attn_mask,                       # ⬅︎ 새 인자
        #     ).squeeze(-1)

        #     t2i_cls = self.fusion_module(                    # ▸ Query = 텍스트
        #         torch.cat([text_emb_g.unsqueeze(1), txt_emb_l_], dim=1),
        #         img_emb_g,
        #         attn_mask = attn_mask.transpose(0, 2),       # 열/행 바꿔 재사용
        #     ).squeeze(-1)

        #     return img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents, i2t_cls, t2i_cls
        
        img_emb_l_ = img_emb_l.view(img_emb_l.size(0), img_emb_l.size(1), -1) # [512, 768, 14, 14] -> [512, 768, 196]
        img_emb_l_ = img_emb_l_.permute(0, 2, 1) #patch_num b dim # [512, 196, 768]
        
        text_emb_l_ = text_emb_l.view(text_emb_l.size(0), text_emb_l.size(1), -1)
        text_emb_l_ = text_emb_l_.permute(0, 2, 1) #patch_num b dim # [97, 512, 768]

        #image_features (batch_size,patch_num,dim)
        #text_features (query_num,dim)
        # 

        if feat :
            t2i_logit, atten_t2i, t2i_feat = self.t2i_fusion_module(torch.cat([img_emb_g.unsqueeze(1) , img_emb_l_], dim=1) , text_emb_g, return_feat=True)
            i2t_logit, atten_i2t, i2t_feat = self.i2t_fusion_module(torch.cat([text_emb_g.unsqueeze(1) , text_emb_l_], dim=1) , img_emb_g, return_feat=True) #ver2.1
            # i2t_logit, atten_i2t, i2t_feat = self.i2t_fusion_module(text_emb_l_ , img_emb_g, return_feat=True) # ver2.2
            # i2t_logit, atten_i2t, i2t_feat = self.i2t_fusion_module(text_emb_g.unsqueeze(1) , img_emb_g, return_feat=True) # ver2.3
            
            # text_emb_g_ = text_emb_g.detach()
            # text_emb_g_unsq = text_emb_g_.unsqueeze(1).expand(-1, text_emb_l_.size(1), -1)  # [B, L-1, D]
            # text_emb_l_cat = torch.cat([text_emb_g_unsq, text_emb_l_[:,:,:]], dim=-1)  # [B, L-1, 2D]
            # text_emb_l_fused = self.i2t_cls_fuse_mlp(text_emb_l_cat)  # [B, L-1, D]
            # text_emb_gate = torch.sigmoid(self.i2t_cls_local_gating(text_emb_l_cat))  # [B, L-1, D]
            # text_emb_l_gated = text_emb_gate*text_emb_l_ + (1 - text_emb_gate)*text_emb_l_fused  # [B, L-1, D]
            
            #i2t_cls, atten_i2t, i2t_feat = self.i2t_fusion_module(text_emb_l_gated, img_emb_g, return_feat=True)
            
            i2t_logit = i2t_logit.squeeze(-1)
            t2i_logit = t2i_logit.squeeze(-1)
            
            # i2t_cls = i2t_cls*torch.exp(self.i2t_tau)
            # t2i_cls = t2i_cls*torch.exp(self.t2i_tau)
            
            return img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents, i2t_logit, t2i_logit, atten_i2t, atten_t2i, i2t_feat, t2i_feat
        
        t2i_logit = self.t2i_fusion_module(torch.cat([img_emb_g.unsqueeze(1) , img_emb_l_], dim=1) , text_emb_g).squeeze(-1) # query shape : (B, 1, D), Key shape : (B, S_img+1, D), Value shape : (B, S_img+1, D)

        #pos=self.img_encoder.model.pos_embed[:, 1:, :].transpose(0,1)

        # text to image

        # 
        i2t_logit = self.i2t_fusion_module(torch.cat([text_emb_g.unsqueeze(1) , text_emb_l_], dim=1) , img_emb_g).squeeze(-1) # ver2.1
        #i2t_logit = self.i2t_fusion_module(text_emb_l_, img_emb_g).squeeze(-1) # ver2.2
        #i2t_logit = self.i2t_fusion_module(text_emb_g.unsqueeze(1), img_emb_g).squeeze(-1) # ver2.3
        # text_emb_g_ = text_emb_g.detach()
        # text_emb_g_unsq = text_emb_g_.unsqueeze(1).expand(-1, text_emb_l_.size(1), -1)  # [B, L-1, D]
        # text_emb_l_cat = torch.cat([text_emb_g_unsq, text_emb_l_[:,:,:]], dim=-1)  # [B, L-1, 2D]
        # text_emb_l_fused = self.i2t_cls_fuse_mlp(text_emb_l_cat)  # [B, L-1, D]
        # text_emb_gate = torch.sigmoid(self.i2t_cls_local_gating(text_emb_l_cat))  # [B, L-1, D]
        # text_emb_l_gated = text_emb_gate*text_emb_l_ + (1 - text_emb_gate)*text_emb_l_fused  # [B, L-1, D]
        
        #i2t_cls = self.i2t_fusion_module(text_emb_l_gated, img_emb_g).squeeze(-1) # ver2.4
        
        # i2t_cls = i2t_cls*torch.exp(self.i2t_tau)
        # t2i_cls = t2i_cls*torch.exp(self.t2i_tau)
        # 
        return img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents, i2t_logit, t2i_logit
    
    def i2t_mcq_forward(self, x, i2t_only=False):
        # img encoder branch
        img_emb_l, img_emb_g = self.image_encoder_forward(x["imgs"])

        B, N, L = x["caption_ids"].shape
        
        caption_ids = x["caption_ids"].view(B * N, L)
        attention_mask = x["attention_mask"].view(B * N, L)
        token_type_ids = x["token_type_ids"].view(B * N, L)
        
        # text encorder branch
        text_emb_l, text_emb_g, sents = self.text_encoder_forward(
            caption_ids, attention_mask, token_type_ids
        ) # text_emb_l: (B*4, D, T), text_emb_g: (B*4, D)
        
        D = text_emb_g.shape[-1]
        
        text_emb_l = text_emb_l.view(B, N, text_emb_l.size(1), text_emb_l.size(2))  # (B, N, D, T)
        text_emb_g = text_emb_g.view(B, N, text_emb_g.size(1))  # (B, N, D)
        sents = [ sents[i] for i in range(0, len(sents), N) ]  # (B, )
        
        img_emb_l_ = img_emb_l.view(img_emb_l.size(0), img_emb_l.size(1), -1) # (B, D, H, W) -> (B, D, 1+S_img)
        img_emb_l_ = img_emb_l_.permute(0, 2, 1) #patch_num b dim # (B, 1+S_img, D)
        text_emb_l_ = text_emb_l.permute(0, 1, 3, 2)  # (B, N, T, D)
        
        img_emb_ = torch.cat([img_emb_g.unsqueeze(1) , img_emb_l_], dim=1)  # (B, 1+S_img, D)
        text_emb_ = torch.cat([text_emb_g.unsqueeze(2) , text_emb_l_], dim=2)  # (B, N, 1+S_txt, D)
        
        img_emb_g_ = img_emb_g.unsqueeze(1)  # (B, 1, D)
        img_emb_g_ = img_emb_g_.unsqueeze(1).expand(B, N, 1, D)  # (B, N, 1, D)
        img_emb_g_ = img_emb_g_.reshape(B * N, 1, D)  # (B*N, 1, D)
        img_emb_g_ = img_emb_g_.permute(1, 0, 2) # (1, B*N, D)
        
        text_emb_g_ = text_emb_g.reshape(B * N, 1, D) # (B*N, 1, D)
        text_emb_g_ = text_emb_g_.permute(1, 0, 2)  # (1, B*N, D)

        B, S_img, D = img_emb_.shape
        _, N, S_txt, _ = text_emb_.shape
        
        # (1) 이미지 토큰 memory 준비
        # img_emb_l_: (B, S_img, D) → (S_img, B, D)
        img_emb_ = img_emb_.permute(1, 0, 2)               # (S_img+1, B, D)

        # 이미지 token을 보기 N과 pair 만들기 위해 batch 확장:
        # (S_img, B, 1, D) → (S_img, B, N, D) → (S_img, B*N, D)
        img_emb_ = img_emb_.unsqueeze(2).expand(S_img, B, N, D) # (S_img, B, N, D)
        img_emb_ = img_emb_.reshape(S_img, B * N, D)       # (S_img, B*N, D)
        img_emb_ = img_emb_.transpose(0,1)  # (B*N, S_img, D)
        
        text_emb_ = text_emb_.permute(2,0,1,3) # (S_txt, B, N, D)
        text_emb_ = text_emb_.reshape(S_txt, B * N, D) # (S_txt, B*N, D)
        text_emb_ = text_emb_.transpose(0,1)  # (B*N, S_txt, D)
        
        if i2t_only:
            t2i_logit = None
        else :
            t2i_logit = self.t2i_fusion_module(img_emb_, text_emb_g_, inside_repeat=False).squeeze(-1).squeeze(-1) # (B*N, S_img, D), (1, B*N, D) -> (B*N, 1, 1) -> (B*N)
            t2i_logit = t2i_logit.view(B, N)
        #    t2i_logit = t2i_logit*torch.exp(self.t2i_tau)
        i2t_logit = self.i2t_fusion_module(text_emb_, img_emb_g_, inside_repeat=False).squeeze(-1).squeeze(-1) # (B*N, S_txt, D), (1, B*N, D) -> (B*N, 1, 1) -> (B*N)
        i2t_logit = i2t_logit.view(B, N)
        #i2t_logit = i2t_logit*torch.exp(self.i2t_tau)
        #ver2.1

        #i2t_cls = self.i2t_fusion_module(text_emb_l_, img_emb_g).squeeze(-1) # ver2.2
        #i2t_cls = self.i2t_fusion_module(text_emb_g.unsqueeze(1), img_emb_g).squeeze(-1) # ver2.3
        # text_emb_g_ = text_emb_g.detach()
        # text_emb_g_unsq = text_emb_g_.unsqueeze(1).expand(-1, text_emb_l_.size(1), -1)  # [B, L-1, D]
        # text_emb_l_cat = torch.cat([text_emb_g_unsq, text_emb_l_[:,:,:]], dim=-1)  # [B, L-1, 2D]
        # text_emb_l_fused = self.i2t_cls_fuse_mlp(text_emb_l_cat)  # [B, L-1, D]
        # text_emb_gate = torch.sigmoid(self.i2t_cls_local_gating(text_emb_l_cat))  # [B, L-1, D]
        # text_emb_l_gated = text_emb_gate*text_emb_l_ + (1 - text_emb_gate)*text_emb_l_fused  # [B, L-1, D]
        
        #i2t_cls = self.i2t_fusion_module(text_emb_l_gated, img_emb_g).squeeze(-1) # ver2.4
                        
        return i2t_logit, t2i_logit
    
    def t2i_mcq_forward(self, x, t2i_only=False):
        """
        Text-to-Image MCQ forward
        x["imgs"]        : (B, N, C, H, W)  # 각 텍스트 질문마다 이미지 보기 N개
        x["caption_ids"] : (B, N, L)        # 같은 텍스트가 N번 반복된 형태
        """

        # ---------------------------
        # 1) 이미지 / 텍스트 인코딩
        # ---------------------------
        # 이미지: (B, N, C, H, W) -> (B*N, C, H, W)
        imgs = x["imgs"]
        B, N, C, H, W = imgs.shape
        imgs_flat = imgs.view(B * N, C, H, W)

        img_emb_l, img_emb_g = self.image_encoder_forward(imgs_flat)
        # img_emb_l: (B*N, D, Hf, Wf), img_emb_g: (B*N, D)

        # 텍스트: (B, N, L) -> (B*N, L)
        B_t, N_t, L = x["caption_ids"].shape
        assert B_t == B and N_t == N, "MCQ batch 차원 불일치"

        caption_ids = x["caption_ids"].reshape(B * N, L) # (B*N, L)
        attention_mask = x["attention_mask"].reshape(B * N, L) # (B*N, L)
        token_type_ids = x["token_type_ids"] # (B*N, L)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape(B * N, L)
        text_emb_l, text_emb_g, sents = self.text_encoder_forward(
            caption_ids, attention_mask, token_type_ids
        )
        # text_emb_l: (B*N, D, T), text_emb_g: (B*N, D)

        D = text_emb_g.shape[-1]

        # -------------------------------------------------
        # 2) (B, N, ...) 형태로 재배치 + 문장 리스트 정리
        #    - 이미지: (B, N, D, Hf, Wf), (B, N, D)
        #    - 텍스트: (B, N, D, T), (B, N, D) → 같은 텍스트이므로 N 중 첫 번째만 사용
        # -------------------------------------------------
        # 이미지 로컬 임베딩
        _, D_img, Hf, Wf = img_emb_l.shape
        img_emb_l_bn = img_emb_l.view(B, N, D_img, Hf, Wf)    # (B, N, D, Hf, Wf)
        img_emb_g_bn = img_emb_g.view(B, N, D_img)            # (B, N, D)

        # 텍스트 로컬 임베딩
        _, D_txt, T = text_emb_l.shape
        text_emb_l_bn = text_emb_l.view(B, N, D_txt, T)       # (B, N, D, T)
        text_emb_g_bn = text_emb_g.view(B, N, D_txt)          # (B, N, D)

        # sents 도 (B,) 개로 줄이기 (각 질문마다 첫 번째 것만 사용)
        sents = [sents[i] for i in range(0, len(sents), N)]   # 길이 B list

        # -------------------------------------------------
        # 3) Fusion module 입력을 위한 pair 단위(B*N) 시퀀스 구성
        #    - 각 pair = (질문 b, 보기 이미지 n)
        # -------------------------------------------------
        # 3-1) 이미지 토큰 시퀀스: [global; local_patches]
        img_emb_l_flat = img_emb_l_bn.view(B * N, D_img, -1)      # (B*N, D, S_img)
        img_emb_l_flat = img_emb_l_flat.permute(0, 2, 1)          # (B*N, S_img, D)
        img_emb_g_pair = img_emb_g_bn.view(B * N, D_img)          # (B*N, D)

        img_emb_ = torch.cat([img_emb_g_pair.unsqueeze(1), img_emb_l_flat], dim=1) # (B*N, 1 + S_img, D)

        # 3-2) 텍스트 토큰 시퀀스: [global; local_tokens]
        text_emb_l_flat = text_emb_l_bn.view(B * N, D_txt, T)     # (B*N, D, T)
        text_emb_l_flat = text_emb_l_flat.permute(0, 2, 1)        # (B*N, T, D)
        text_emb_g_pair = text_emb_g_bn.view(B * N, D_txt)        # (B*N, D)

        # (B*N, 1 + S_txt, D)
        text_emb_ = torch.cat([text_emb_g_pair.unsqueeze(1), text_emb_l_flat], dim=1)

        # -------------------------------------------------
        # 4) Fusion module용 global query tensor 형태 맞추기
        #    (1, B*N, D) : i2t_mcq_forward 와 동일한 인터페이스 유지
        # -------------------------------------------------
        img_emb_g_ = img_emb_g_pair.view(B * N, 1, D).permute(1, 0, 2)   # (1, B*N, D)
        text_emb_g_ = text_emb_g_pair.view(B * N, 1, D).permute(1, 0, 2) # (1, B*N, D)

        # -------------------------------------------------
        # 5) Fusion module 계산
        #    - t2i: (image tokens, text global)
        #    - i2t: (text tokens, image global)
        # -------------------------------------------------
        t2i_logit = self.t2i_fusion_module(
            img_emb_, text_emb_g_, inside_repeat=False
        ).squeeze(-1).squeeze(-1)   # (B*N, 1+S_img, D) * (1, B*N, D) -> (B*N,)
        t2i_logit = t2i_logit.view(B, N)
        
        if t2i_only:
            i2t_logit = None
        else:
            i2t_logit = self.i2t_fusion_module(
                text_emb_, img_emb_g_, inside_repeat=False
            ).squeeze(-1).squeeze(-1)   # (B*N, 1+S_txt, D) * (1, B*N, D) -> (B*N,)
            i2t_logit = i2t_logit.view(B, N)
            #i2t_logit = i2t_logit*torch.exp(self.i2t_tau)
        
       # t2i_logit = t2i_logit*torch.exp(self.t2i_tau)
        # -------------------------------------------------
        # 6) 반환:
        #    - 이미지 임베딩: (B, N, D, Hf, Wf), (B, N, D)
        #    - 텍스트 임베딩: (B, D, T), (B, D)
        #    - sents      : 길이 B list
        #    - i2t_logit  : (B, N)
        #    - t2i_logit  : (B, N)
        # -------------------------------------------------
        return i2t_logit, t2i_logit
        

    def get_global_similarities(self, img_emb_g, text_emb_g):
        img_emb_g = img_emb_g.detach().cpu().numpy()
        text_emb_g = text_emb_g.detach().cpu().numpy()
        global_similarities = metrics.pairwise.cosine_similarity(img_emb_g, text_emb_g)
        global_similarities = torch.Tensor(global_similarities)
        return global_similarities

    def get_local_similarities(self, img_emb_l, text_emb_l, cap_lens):
        batch_size = img_emb_l.shape[0]
        similarities = []
        for i in range(len(text_emb_l)):
            words_num = cap_lens[i]
            word = (
                text_emb_l[i, :, 1 : words_num + 1].unsqueeze(0).contiguous()
            )  # [1, 768, 25]

            word = word.repeat(batch_size, 1, 1)  # [48, 768, 25]
            context = img_emb_l  # [48, 768, 19, 19]

            weiContext, attn = loss.CARZero_loss.attention_fn(
                word, context, 4.0
            )  # [48, 768, 25], [48, 25, 19, 19]

            word = word.transpose(1, 2).contiguous()  # [48, 25, 768]
            weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 25, 768]

            word = word.view(batch_size * words_num, -1)  # [1200, 768]
            weiContext = weiContext.view(batch_size * words_num, -1)  # [1200, 768]
            #
            row_sim = loss.CARZero_loss.cosine_similarity(word, weiContext)
            row_sim = row_sim.view(batch_size, words_num)  # [48, 25]

            row_sim.mul_(5.0).exp_()
            row_sim, max_row_idx = torch.max(row_sim, dim=1, keepdim=True)  # [48, 1]

            row_sim = torch.log(row_sim)

            similarities.append(row_sim)

        local_similarities = torch.cat(similarities, 1).detach().cpu()

        return local_similarities

    def get_attn_maps(self, img_emb_l, text_emb_l, sents, temp1=4.0, max_len=6):
        batch_size,t,h,w = img_emb_l.shape

        cap_lens = [len([w for w in sent if not w.startswith("[")]) + 1 for sent in sents]
        attn_maps = []
        # cap_lens = cap_lens.data.tolist()
        for i in range(text_emb_l.shape[0]):

            # Get the i-th text description
            words_num = cap_lens[i]  # 10
            sep_idx = next((idx for idx, token in enumerate(sents[i]) if token == "[SEP]"), text_emb_l.shape[2])
            word_end = min(words_num, sep_idx)
            word = text_emb_l[i, :, 1:word_end].unsqueeze(0).contiguous()  # [1, 768, 25]
            word = word.repeat(batch_size, 1, 1)  # [48, 768, 25]
            context = img_emb_l  # [48, 768, 19, 19]

            weiContext, attn = loss.CARZero_loss.attention_fn(
                word, context, temp1
            )  # [48, 768, 25], [48, 25, 19, 19]

            attn_maps.append(
                attn.contiguous().detach().cpu().numpy()
            )  # add attention for curr index  [25, 19, 19]
        return attn_maps

    def plot_attn_maps(self, attn_maps, imgs, sents, epoch_idx=0, batch_idx=0):

        img_set, _ = utils.build_attention_images(
            imgs,
            attn_maps,
            max_word_num=self.cfg.data.text.word_num,
            nvis=self.cfg.train.nvis,
            rand_vis=self.cfg.train.rand_vis,
            sentences=sents,
        )

        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = (
                f"{self.cfg.output_dir}/"
                f"attention_maps_epoch{epoch_idx}_"
                f"{batch_idx}.png"
            )
            im.save(fullpath)

    def process_text(self, text, device):
        if type(text) == str:
            text = [text]

        processed_text_tensors = []
        for t in text:
            # use space instead of newline
            t = t.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(t)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            all_sents = []

            for t in captions:
                t = t.replace("\ufffd\ufffd", " ")
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(t.lower())

                if len(tokens) <= 1:
                    continue

                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                all_sents.append(" ".join(included_tokens))

            t = " ".join(all_sents)

            text_tensors = self.tokenizer(
                t,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.cfg.data.text.word_num,
            )
            text_tensors["sent"] = [
                self.ixtoword[ix] for ix in text_tensors["input_ids"][0].tolist()
            ]
            processed_text_tensors.append(text_tensors)

        caption_ids = torch.stack([x["input_ids"] for x in processed_text_tensors])
        attention_mask = torch.stack(
            [x["attention_mask"] for x in processed_text_tensors]
        )
        token_type_ids = torch.stack(
            [x["token_type_ids"] for x in processed_text_tensors]
        )

        if len(text) == 1:
            caption_ids = caption_ids.squeeze(0).to(device)
            attention_mask = attention_mask.squeeze(0).to(device)
            token_type_ids = token_type_ids.squeeze(0).to(device)
        else:
            caption_ids = caption_ids.squeeze().to(device)
            attention_mask = attention_mask.squeeze().to(device)
            token_type_ids = token_type_ids.squeeze().to(device)

        cap_lens = []
        for txt in text:
            cap_lens.append(len([w for w in txt if not w.startswith("[")]))

        return {
            "caption_ids": caption_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "cap_lens": cap_lens,
        }

    def process_class_prompts(self, class_prompts, device):
        cls_2_processed_txt = {}
        for k, v in class_prompts.items():
            cls_2_processed_txt[k] = self.process_text(v, device)

        return cls_2_processed_txt

    def process_img(self, paths, device, augmentation="test", corrupt=None, severity=None, corrupt_type=None):
        transform = builder.build_transformation(self.cfg, split=augmentation)
        if type(paths) == str:
            paths = [paths]

        all_imgs = []
        for p in paths:
            x = cv2.imread(str(p), 0)
            if x is None:
                print(f"[ERROR] Failed to read image: {p}", flush=True)
                continue  # skip invalid image
            if corrupt :
                #print(f"[INFO] Applying corruption: {corrupt_type} with severity {severity} to image: {p}", flush=True)
                x = Image.fromarray(x).convert("L")
                x = corrupt(x, severity, corrupt_type)
                #x.save("temp_corrupted.png")
                x = np.array(x)

            x = self._resize_img(x, self.cfg.data.image.imsize)
            img = Image.fromarray(x).convert("RGB")
            img = transform(img)

            # 안전하게 텐서로 처리
            if isinstance(img, torch.Tensor):
                all_imgs.append(img.clone().detach())
            else:
                all_imgs.append(torch.tensor(img))

        if len(all_imgs) == 0:
            raise RuntimeError("No valid images found in batch!")

        all_imgs = torch.stack(all_imgs).to(device)
        return all_imgs
    
    def process_single_img(self, paths):

        transform = builder.build_transformation(self.cfg, split="test")
        x = cv2.imread(str(paths), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")
        img = transform(img)

        return img

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
