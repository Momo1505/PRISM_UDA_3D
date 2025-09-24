# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Support for seg_weight and forward_with_aux
# - Update debug output system

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from ..utils.dacs_transforms import get_mean_std
from ..utils.visualization import prepare_debug_out, subplotimg
from .base import BaseSegmentor
from ..decode_heads.unetr import UNETR


@SEGMENTORS.register_module()
class EncoderDecoder3D(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                     img_shape=(256, 256, 40),
                     input_dim=1,
                     output_dim=2,
                     embed_dim=768,
                     patch_size=16,
                     num_heads=12,
                     dropout=0.1,
                     train_cfg=None,
                     test_cfg=None,
                     pretrained=None,
                     init_cfg=None,**kwargs):
        super().__init__(init_cfg)
        self.backbone = UNETR(
                img_shape=img_shape,
                input_dim=input_dim,
                output_dim=output_dim,
                embed_dim=embed_dim,
                patch_size=patch_size,
                num_heads=num_heads,
                dropout=dropout
            )
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.automatic_debug = True
        self.debug = False
        self.debug_output = {}
        self.local_iter = 0
        if train_cfg is not None and 'log_config' in train_cfg:
            self.debug_img_interval = train_cfg['log_config']['img_interval']

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        return x

    def generate_pseudo_label(self, img, img_metas):
        self.update_debug_state()
        
        out = self.backbone(img)
        return out

    def encode_decode(self, img, img_metas, upscale_pred=True):
        """Encode images with backbone and decode into a semantic segmentation map."""
        return self.backbone(img)

    def forward_with_aux(self, img, img_metas):
        self.update_debug_state()
        ret = {}
        out = self.extract_feat(img)
        ret['main'] = out
        return ret

    def _decode_head_forward_train(self,
                                   x,
                                   img_metas,
                                   gt_semantic_seg,
                                   seg_weight=None,
                                   return_logits=False):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.backbone.forward_train(x,gt_semantic_seg,seg_weight)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.backbone.forward(x)
        return seg_logits


    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def update_debug_state(self):
        self.debug_output = {}
        if self.automatic_debug:
            self.debug = (self.local_iter % self.debug_img_interval == 0)
        

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      seg_weight=None,
                      return_feat=False,
                      return_logits=False):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        self.update_debug_state()

        losses = dict()
        loss_decode = self._decode_head_forward_train(img, img_metas,
                                                      gt_semantic_seg,
                                                      seg_weight,
                                                      return_logits)
        losses.update(loss_decode)

        self.local_iter += 1
        return losses


    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""
        return self.backbone(img)

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style."""
        seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        
        flip = img_meta[0].get('flip', False)
        if flip:
            axis_map = {'horizontal': 4, 'vertical': 3, 'depth': 2}  # For 5D tensor (B,C,D,H,W)
            flip_direction = img_meta[0]['flip_direction']
            if flip_direction in axis_map:
                output = output.flip(dims=(axis_map[flip_direction],))
        
        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        return list(seg_pred)

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
