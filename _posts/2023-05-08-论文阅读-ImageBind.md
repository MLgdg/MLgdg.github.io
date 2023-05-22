---
layout:     post
title:      论文阅读-ImageBind
subtitle:   
date:       2023-05-04
author:     高庆东
header-img: img/ceshi.jpg
catalog: true
tags:
    - 多模态
    - audio
---


## 模型结构
![imagebind](/img/20230313/imagebind.png)  

1、6个Transformer 分别对应六种模态， 
2、每个模态有个hear  
3、对每种模态输出加一个参数，这个参数控制占比   
4、核心思想是对齐模态都和图像模态对齐，图片和文本模态使用CLIP对模型进行初始化  
其他模态和图像进行对齐训练，所用模态都和图像模态对齐，这样就对齐了所有模态

## video模态
使用clip初始化，vit模型，不同的是输入的是video 使用3d卷积切片  
使用绝对位置编码，其他tirck没有，3d卷积后展开方式和2d一样拉平  
视频和图像的处理完全一致，时间维度(帧)直接拉平，位置编码的长度THW  
B C (T) H W -> B T\*H\*W C  
使用3d卷积展开   

"""
class PatchEmbedGeneric(nn.Module):

    def __init__(self, proj_stem, norm_layer: Optional[nn.Module] = None):
        super().__init__()

        if len(proj_stem) > 1:
            self.proj = nn.Sequential(*proj_stem)
        else:
            # Special case to be able to load pre-trained models that were
            # trained with a standard stem
            self.proj = proj_stem[0]
        self.norm_layer = norm_layer

    def get_patch_layout(self, img_size):
        with torch.no_grad():
            dummy_img = torch.zeros(
                [
                    1,
                ]
                + img_size
            )
            dummy_out = self.proj(dummy_img)
        embed_dim = dummy_out.shape[1]
        patches_layout = tuple(dummy_out.shape[2:])
        num_patches = np.prod(patches_layout)
        return patches_layout, num_patches, embed_dim

    def forward(self, x):
        x = self.proj(x)
        # B C (T) H W -> B (T)HW C
        x = x.flatten(2).transpose(1, 2)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return x
"""

## audio模态
对音频分段然后提取mel特征，使用2d卷积扩展

## text模态
自回归模型
## depth模态
## 红外模态
## 坐标模态
