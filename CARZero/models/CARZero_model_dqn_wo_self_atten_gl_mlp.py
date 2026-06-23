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
        self.fusion_module = builder.build_dqn_wo_self_atten_mlp_orig_module(cfg)

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
            t2i_cls, atten_t2i, t2i_feat = self.fusion_module(torch.cat([img_emb_g.unsqueeze(1) , img_emb_l_], dim=1) , text_emb_g, return_feat=True) # ()
            i2t_cls, atten_i2t, i2t_feat = self.fusion_module(torch.cat([text_emb_g.unsqueeze(1) , text_emb_l_], dim=1) , img_emb_g, return_feat=True)
            
            i2t_cls = i2t_cls.squeeze(-1)
            t2i_cls = t2i_cls.squeeze(-1)
            return img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents, i2t_cls, t2i_cls, atten_i2t, atten_t2i, i2t_feat, t2i_feat
        
        t2i_cls = self.fusion_module(torch.cat([img_emb_g.unsqueeze(1) , img_emb_l_], dim=1) , text_emb_g).squeeze(-1)

        #pos=self.img_encoder.model.pos_embed[:, 1:, :].transpose(0,1)

        # text to image

        # 
        i2t_cls = self.fusion_module(torch.cat([text_emb_g.unsqueeze(1) , text_emb_l_], dim=1) , img_emb_g).squeeze(-1)
        # 
        return img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents, i2t_cls, t2i_cls

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
            # TODO: remove [SEP]
            # word = text_emb_l[i, :, 1:words_num+1].unsqueeze(0).contiguous()    # [1, 768, 25]
            word = text_emb_l[i, :, 1:words_num].unsqueeze(0).contiguous()  # [1, 768, 25]
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
