import torch
import torch.nn as nn
import torch.nn.functional as torch_functional

from stdc_nets import STDCNet1446, STDCNet813

class ConvBNReLU(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size= 3, stride= 1, padding= 1, *args, **kwargs) -> None:
        super(ConvBNReLU, self).__init__(*args, **kwargs)

        self.conv = nn.Conv2d(
            in_channels= channels_in,
            out_channels= channels_out,
            kernel_size= kernel_size,
            stride= stride,
            padding= padding,
            bias= False,
            )
        
        self.bn = nn.BatchNorm2d(channels_out)
        #self.bn = nn.BatchNorm2d(channels_out, activate= 'none')

        self.relu = nn.ReLU()

        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a= 1)
                if not layer.bias is None: nn.init.constant(layer.bias, 0)


class BiSeNetOutput(nn.Module):
    def __init__(self, channels_in, channels_mid, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(channels_in, channels_mid, kernel_size= 3, stride= 1, padding= 1)
        self.conv_out = nn.Conv2d(channels_mid, n_classes, kernel_size= 1, bias= False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a= 1)
                if not layer.bias is None: nn.init.constant_(layer.bias, 0)

    def get_params(self):
        weight_decay_params, no_weight_decay_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight_decay_params.append(module.weight)
                if not module.bias is None:
                    no_weight_decay_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                no_weight_decay_params += list(module.parameters())
        return weight_decay_params, no_weight_decay_params
    

class AttentionRefinementModule(nn.Module):
    def __init__(self, channels_in, channels_out, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(channels_in, channels_out, kernel_size= 3, stride= 1, padding= 1)
        self.conv_attention = nn.Conv2d(channels_out, channels_out, kernel_size= 1, bias= False)
        self.bn_attention = nn.BatchNorm2d(channels_out)
        #self.bn_attention = nn.BatchNorm2d(channels_out, activation= 'none')
        self.sigmoid_attention = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)

        #attention = nn.functional.avg_pool2d(feat, feat.size()[2:])

        size_array = [int(s) for s in feat.size()[2:]]
        attention = torch_functional.avg_pool2d(feat, size_array)
        attention = self.conv_attention(attention)
        attention = self.bn_attention(attention)
        attention = self.sigmoid_attention(attention)
        out = torch.mul(feat, attention)
        return out

    def init_weight(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a= 1)
                if not layer.bias is None: nn.init.constant_(layer.bias, 0)


class ContextPath(nn.Module):
    def __init__(self, backbone_name= 'No Net Selected', pretrain_model= '', use_conv_last= False, input_size= 512, *args, **kwargs):
        super(ContextPath, self).__init__(*args, **kwargs)
        
        self.backbone_name = backbone_name
        self.input_size = input_size
        print('backbone: ', backbone_name)

        if backbone_name == 'STDCNet1446':
            self.backbone = STDCNet1446(pretrain_model= pretrain_model, use_conv_last= use_conv_last)
            self.attention_refinement_module16 = AttentionRefinementModule(channels_in= 512, channels_out= 128)
            channels_in = 1024 #inplanes?
            #if use_conv_last: channels_in = 1024
            self.attention_refinement_module32 = AttentionRefinementModule(channels_in, channels_out= 128)
            self.conv_head32 = ConvBNReLU(128, 128, kernel_sizes= 3, stride= 1, padding= 1)
            self.conv_head16 = ConvBNReLU(128, 128, kernel_sizes= 3, stride= 1, padding= 1)
            self.conv_avg = ConvBNReLU(channels_in, 128, kernel_sizes= 1, stride= 1, padding= 0)
        
        elif backbone_name == 'STDCNet813':
            self.backbone = STDCNet813(pretrain_model= pretrain_model, use_conv_last= use_conv_last)
            self.attention_refinement_module16 = AttentionRefinementModule(512, 128)
            channels_in = 1024 #inplanes?
            #if use_conv_last: channels_in = 1024
            self.attention_refinement_module32 = AttentionRefinementModule(channels_in, 128)
            self.conv_head32 = ConvBNReLU(128, 128, kernel_size= 3, stride= 1, padding= 1)
            self.conv_head16 = ConvBNReLU(128, 128, kernel_size= 3, stride= 1, padding= 1)
            self.conv_avg = ConvBNReLU(channels_in, 128, kernel_size= 1, stride= 1, padding= 0)

        else:
            print("backbone is not in backbone lists")
            exit(0)

        if self.input_size == 512:
            self.Height8 = torch.tensor(64)
            self.Width8 = torch.tensor(128)

            self.Height16 = torch.tensor(32)
            self.Width16 = torch.tensor(64)

            self.Height32 = torch.tensor(16)
            self.Width32 = torch.tensor(32)

        elif self.input_size == 720:
            self.Height8 = torch.tensor(90)
            self.Width8 = torch.tensor(120)

            self.Height16 = torch.tensor(45)
            self.Width16 = torch.tensor(60)

            self.Height32 = torch.tensor(23)
            self.Width32 = torch.tensor(30)

        elif self.input_size == 768:
            self.Height8 = torch.tensor(96)
            self.Width8 = torch.tensor(192)

            self.Height16 = torch.tensor(48)
            self.Width16 = torch.tensor(96)

            self.Height32 = torch.tensor(24)
            self.Width32 = torch.tensor(48)

        elif self.input_size == 1024:
            self.Height8 = torch.tensor(128)
            self.Width8 = torch.tensor(256)

            self.Height16 = torch.tensor(64)
            self.Width16 = torch.tensor(128)

            self.Height32 = torch.tensor(32)
            self.Width32 = torch.tensor(64)

        else:
            print("input_size is not in input_size lists")
            exit(0)

        self.init_weight()

    def forward(self, x):

        feat2, feat4, feat8, feat16, feat32 = self.backbone(x)
        size_array = [int(s) for s in feat32.size()[2:]]
        avg = torch_functional.avg_pool2d(feat32, size_array)

        avg = self.conv_avg(avg)
        avg_up = torch_functional.interpolate(avg, size= (self.Height32, self.Width32), mode= 'nearest')

        feat32_arm = self.attention_refinement_module32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = torch_functional.interpolate(feat32_sum, size= (self.Height16, self.Width16), mode= 'nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.attention_refinement_module16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = torch_functional.interpolate(feat16_sum, size= (self.Height8, self.Width8), mode= 'nearest')
        feat16_up = self.conv_head16(feat16_up)
        
        return feat2, feat4, feat8, feat16, feat16_up, feat32_up # x8, x16

    def init_weight(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a= 1)
                if not layer.bias is None: nn.init.constant_(layer.bias, 0)

    def get_params(self):
        weight_decay_params, no_weight_decay_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight_decay_params.append(module.weight)
                if not module.bias is None:
                    no_weight_decay_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                no_weight_decay_params += list(module.parameters())
        return weight_decay_params, no_weight_decay_params


class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__(*args, **kwargs)
        self.conv1 = ConvBNReLU(3, 64, kernel_size= 7, stride= 2, padding= 3)
        self.conv2 = ConvBNReLU(64, 64, kernel_size= 3, stride= 2, padding= 1)
        self.conv3 = ConvBNReLU(64, 64, kernel_size= 3, stride= 2, padding= 1)
        self.conv_out = ConvBNReLU(64, 128, kernel_size= 1, stride= 1, padding= 0)
        self.init_weight()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a= 1)
                if not layer.bias is None: nn.init.constant_(layer.bias, 0)

    def get_params(self):
        weight_decay_params, no_weight_decay_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                weight_decay_params.append(module.weight)
                if not module.bias is None:
                    no_weight_decay_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                no_weight_decay_params += list(module.parameters())
        return weight_decay_params, no_weight_decay_params


class FeatureFusionModule(nn.Module):
    def __init__(self, channels_in, channels_out, *args, **kwargs):
        super(FeatureFusionModule, self).__init__(*args, **kwargs)
        self.convblk = ConvBNReLU(
            channels_in,
            channels_out, 
            kernel_size= 1,
            stride= 1,
            padding= 0,
            )
        self.conv1 = nn.Conv2d(
            channels_out,
            channels_out // 4,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            bias = False,
            )
        self.conv2 = nn.Conv2d(
            channels_out // 4,
            channels_out,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            bias = False,
            )
        self.relu = nn.ReLU(inplace= True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, spatial_path_features, context_path_features):
        concatenated_features = torch.cat([spatial_path_features, context_path_features], dim= 1)

        feat = self.convblk(concatenated_features)
        # attention = F.avg_pool2d(feat, feat.size()[2:])

        size_array = [int(s) for s in feat.size()[2:]]
        attention = torch.nn.functional.avg_pool2d(feat, size_array)
        attention = self.conv1(attention)
        attention = self.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        feat_attention = torch.mul(feat, attention)
        feat_out = feat_attention + feat
        return feat_out

    def init_weight(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a= 1)
                if not layer.bias is None: nn.init.constant_(layer.bias, 0)

    def get_params(self):
        weight_decay_params, no_weight_decay_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight_decay_params.append(module.weight)
                if not module.bias is None:
                    no_weight_decay_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                no_weight_decay_params += list(module.parameters())
        return weight_decay_params, no_weight_decay_params


class BiSeNet(nn.Module):
    def __init__(self, backbone_name, n_classes, pretrain_model= '', use_boundary_2= False, use_boundary_4= False, use_boundary_8= False, use_boundary_16= False, input_size= 512, use_conv_last= False, heat_map= False, *args, **kwargs):
        super(BiSeNet, self).__init__(*args, **kwargs)
        
        self.use_boundary_2 = use_boundary_2
        self.use_boundary_4 = use_boundary_4
        self.use_boundary_8 = use_boundary_8
        self.use_boundary_16 = use_boundary_16
        self.input_size = input_size

        print('BiSeNet backbone: ', backbone_name)
        self.cp = ContextPath(backbone_name, pretrain_model, input_size= self.input_size, use_conv_last= use_conv_last)
        
        if backbone_name == 'STDCNet1446':
            conv_out_channels_in = 128
            sp2_channels_in = 32
            sp4_channels_in = 64
            sp8_channels_in = 256
            sp16_channels_in = 512
            channels_in = sp8_channels_in + conv_out_channels_in

        elif backbone_name == 'STDCNet813':
            conv_out_channels_in = 128
            sp2_channels_in = 32
            sp4_channels_in = 64
            sp8_channels_in = 256
            sp16_channels_in = 512
            channels_in = sp8_channels_in + conv_out_channels_in

        else:
            print("backbone is not in backbone lists")
            exit(0)

        self.ffm = FeatureFusionModule(channels_in, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(conv_out_channels_in, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(conv_out_channels_in, 64, n_classes)
        
        self.conv_out_sp16 = BiSeNetOutput(sp16_channels_in, 64, 1)
        self.conv_out_sp8 = BiSeNetOutput(sp8_channels_in, 64, 1)
        self.conv_out_sp4 = BiSeNetOutput(sp4_channels_in, 64, 1)
        self.conv_out_sp2 = BiSeNetOutput(sp2_channels_in, 64, 1)

        if self.input_size == 512:
            self.Height = torch.tensor(512)
            self.Width = torch.tensor(1024)
        elif self.input_size == 768:
            self.Height = torch.tensor(768)
            self.Width = torch.tensor(1536)
        elif self.input_size == 1024:
            self.Height = torch.tensor(1024)
            self.Width = torch.tensor(2048)
        elif self.input_size == 720:
            self.Height = torch.tensor(720)
            self.Width = torch.tensor(960)
        else:
            print("input_size is not in input_size lists")
            exit(0)
        
        self.init_weight()

    def forward(self, x):
        # Height, Width = x.size()[2:]
        
        feat_res2, feat_res4, feat_res8, feat_res16, feat_cp8, feat_cp16 = self.cp(x)
        # 16, 24, 40, 112, 
  
        feat_out_sp8 = self.conv_out_sp8(feat_res8)

        feat_out_sp16 = self.conv_out_sp16(feat_res16)

        feat_fuse = self.ffm(feat_res8, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = torch_functional.interpolate(feat_out, size= (self.Height, self.Width), mode= 'nearest')
        feat_out16 = torch_functional.interpolate(feat_out16, size= (self.Height, self.Width), mode= 'nearest')
        feat_out32 = torch_functional.interpolate(feat_out32, size= (self.Height, self.Width), mode= 'nearest')

        return feat_out

    def init_weight(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a= 1)
                if not layer.bias is None: nn.init.constant_(layer.bias, 0)

#lr is learning rate -> lr_multiplier_weight_decay_params should be those that require a different learning rate
    def get_params(self):
        weight_decay_params, no_weight_decay_params, lr_multiplier_weight_decay_params, lr_multiplier_noweight_decay_params = [], [], [], []
        for name, child in self.named_children():
            child_weight_decay_params, child_noweight_decay_params = child.get_params()
            if isinstance(child, (FeatureFusionModule, BiSeNetOutput)):
                lr_multiplier_weight_decay_params += child_weight_decay_params
                lr_multiplier_noweight_decay_params += child_noweight_decay_params
            else:
                weight_decay_params += child_weight_decay_params
                no_weight_decay_params += child_noweight_decay_params
        return weight_decay_params, no_weight_decay_params, lr_multiplier_weight_decay_params, lr_multiplier_noweight_decay_params
