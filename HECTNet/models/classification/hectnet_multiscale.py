from torch import nn, Tensor
import argparse
from typing import Dict, Tuple

from . import register_cls_models
from .base_cls import BaseEncoder
from .config.hectnet_multiscale import get_configuration
from components.layers import ConvLayer, LinearLayer, GlobalPool, Identity
from components.modules import HBlock as Block
from components.modules.cross_attention_fusion import CrossAttentionFusion
from components.modules.multiscale_color_texture import MultiScaleColorTextureGabor


@register_cls_models("hectnet_multiscale")
class HECTNet_MultiScale(BaseEncoder):
    """
    Enhanced HECTNet architecture with multiscale color-texture auxiliary branch
    and cross-attention fusion.
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        num_classes = getattr(opts, "model.classification.n_classes", 101)
        pool_type = getattr(opts, "model.layer.global_pool", "mean")
        
        # Multiscale branch parameters
        aux_dim = getattr(opts, "model.classification.hectnet_multiscale.aux_dim", 32)
        gabor_channels = getattr(opts, "model.classification.hectnet_multiscale.gabor_channels", 16)
        fused_dim = getattr(opts, "model.classification.hectnet_multiscale.fused_dim", 160)

        hectnet_config = get_configuration(opts=opts)
        image_channels = hectnet_config["layer0"]["img_channels"]
        out_channels = hectnet_config["layer0"]["out_channels"]

        super().__init__(*args, **kwargs)

        # store model configuration in a dictionary
        self.model_conf_dict = dict()
        
        # Main backbone - same as original HECTNet
        self.conv_1 = ConvLayer(
            opts=opts,
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            use_norm=True,
            use_act=True,
        )
        self.model_conf_dict["conv1"] = {"in": image_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_1, out_channels = self._make_hblock(
            opts=opts, input_channel=in_channels, cfg=hectnet_config["layer1"]
        )
        self.model_conf_dict["layer1"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_2, out_channels = self._make_hblock(
            opts=opts, input_channel=in_channels, cfg=hectnet_config["layer2"]
        )
        self.model_conf_dict["layer2"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_3, out_channels = self._make_hblock(
            opts=opts, input_channel=in_channels, cfg=hectnet_config["layer3"]
        )
        self.model_conf_dict["layer3"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_4, out_channels = self._make_hblock(
            opts=opts, input_channel=in_channels, cfg=hectnet_config["layer4"],
        )
        self.model_conf_dict["layer4"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_5, out_channels = self._make_hblock(
            opts=opts, input_channel=in_channels, cfg=hectnet_config["layer5"],
        )
        self.model_conf_dict["layer5"] = {"in": in_channels, "out": out_channels}

        self.conv_1x1_exp = Identity()
        self.model_conf_dict["exp_before_cls"] = {
            "in": out_channels,
            "out": out_channels,
        }
        
        # Store main backbone output dimension
        main_dim = out_channels
        
        # Auxiliary multiscale color-texture branch
        self.aux_branch = MultiScaleColorTextureGabor(
            in_channels=image_channels, 
            out_dim=aux_dim, 
            gabor_channels=gabor_channels
        )
        
        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(
            main_dim=main_dim,
            aux_dim=aux_dim,
            fused_dim=fused_dim
        )
        
        # Global pooling for main branch
        self.global_pool = GlobalPool(pool_type=pool_type, keep_dim=False)
        
        # Final classifier using fused features
        self.classifier = LinearLayer(
            in_features=fused_dim, 
            out_features=num_classes, 
            bias=True
        )

        # check model
        self.check_model()

        # weight initialization
        self.reset_parameters(opts=opts)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with multiscale auxiliary branch and cross-attention fusion.
        """
        input_img = x  # Changed from x["image"] to x
        
        # Main backbone forward pass
        out = self.conv_1(input_img)
        out = self.layer_1(out)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.conv_1x1_exp(out)
        
        # Global pooling to get main embedding
        main_embed = self.global_pool(out)  # (B, main_dim)
        
        # Auxiliary branch forward pass
        aux_embed = self.aux_branch(input_img)  # (B, aux_dim)
        
        # Cross-attention fusion
        fused_embed = self.fusion(main_embed, aux_embed)  # (B, fused_dim)
        
        # Final classification
        logits = self.classifier(fused_embed)
        
        return logits  # Changed from {"logits": logits} to logits
    
    def extract_features(self, x: Dict) -> Dict:
        """
        Extract fused features for similarity comparisons or retrieval.
        """
        input_img = x["image"]
        
        # Main backbone forward pass
        out = self.conv_1(input_img)
        out = self.layer_1(out)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.conv_1x1_exp(out)
        
        # Global pooling to get main embedding
        main_embed = self.global_pool(out)
        
        # Auxiliary branch forward pass
        aux_embed = self.aux_branch(input_img)
        
        # Cross-attention fusion
        fused_embed = self.fusion(main_embed, aux_embed)
        
        return {"features": fused_embed}

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        
        # Original HECTNet arguments
        group.add_argument(
            "--model.classification.hectnet_multiscale.attn-dropout",
            type=float,
            default=0.0,
            help="Dropout in attention layer. Defaults to 0.0",
        )
        group.add_argument(
            "--model.classification.hectnet_multiscale.ffn-dropout",
            type=float,
            default=0.0,
            help="Dropout between FFN hectnet. Defaults to 0.0",
        )
        group.add_argument(
            "--model.classification.hectnet_multiscale.dropout",
            type=float,
            default=0.0,
            help="Dropout in attention layer. Defaults to 0.0",
        )
        group.add_argument(
            "--model.classification.hectnet_multiscale.width-multiplier",
            type=float,
            default=1.0,
            help="Width multiplier. Defaults to 1.0",
        )
        group.add_argument(
            "--model.classification.hectnet_multiscale.attn-norm-layer",
            type=str,
            default="layer_norm_2d",
            help="Norm layer in attention block. Defaults to LayerNorm",
        )
        
        # New multiscale branch arguments
        group.add_argument(
            "--model.classification.hectnet_multiscale.aux-dim",
            type=int,
            default=32,
            help="Auxiliary branch output dimension. Defaults to 32",
        )
        group.add_argument(
            "--model.classification.hectnet_multiscale.gabor-channels",
            type=int,
            default=16,
            help="Number of Gabor filter channels. Defaults to 16",
        )
        group.add_argument(
            "--model.classification.hectnet_multiscale.fused-dim",
            type=int,
            default=160,
            help="Fused feature dimension after cross-attention. Defaults to 160",
        )
        
        return parser

    def _make_hblock(
        self, opts, input_channel, cfg: Dict
    ) -> Tuple[nn.Sequential, int]:
        prev_dilation = self.dilation
        block = []

        ffn_multiplier = cfg.get("ffn_multiplier")

        dropout = getattr(opts, "model.classification.hectnet_multiscale.dropout", 0.0)

        block.append(
            Block(
                opts=opts,
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=cfg.get("stride", 1),
                ffn_multiplier=ffn_multiplier,
                n_local_blocks=cfg.get("n_local_blocks", 1),
                n_attn_blocks=cfg.get("n_attn_blocks", 1),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
                dropout=dropout,
                ffn_dropout=getattr(
                    opts, "model.classification.hectnet_multiscale.ffn_dropout", 0.0
                ),
                attn_dropout=getattr(
                    opts, "model.classification.hectnet_multiscale.attn_dropout", 0.0
                ),
                attn_norm_layer=getattr(
                    opts, "model.classification.hectnet_multiscale.attn_norm_layer", "layer_norm_2d"
                ),
                expand_ratio=cfg.get("expand_ratio", 4),
                dilation=prev_dilation,
            )
        )

        input_channel = cfg.get("out_channels")

        return nn.Sequential(*block), input_channel