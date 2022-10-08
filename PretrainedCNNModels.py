import time
import sys
import torch
import torch.nn as nn

sys.path.append('/home/icalixto/tools/pretrained-models.pytorch')
import pretrainedmodels
import pretrainedmodels.utils
from pretrainedmodels.models.torchvision_models import pretrained_settings


class PretrainedDisCNN(object):
    """
    Class that encompasses loading of pre-trained CNN models.
    """
    def __init__(self, pretrained_cnn):
        self.pretrained_cnn = pretrained_cnn
        self.build_load_pretrained_cnn()

    def build_load_pretrained_cnn(self):
        """
            Load a pre-trained CNN using torchvision/cadene.
            Set it into feature extraction mode.
        """
        start = time.time()
        self.load_img = pretrainedmodels.utils.LoadImage()
        image_model_name = self.pretrained_cnn
        self.model = pretrainedmodels.__dict__[image_model_name](num_classes=1000, pretrained='imagenet')
        #self.model.train()
        self.tf_img = pretrainedmodels.utils.TransformImage(self.model)
        # returns features before the application of the last linear transformation
        # in the case of a resnet152, it will be a [1, 2048] tensor
        self.model.last_linear = pretrainedmodels.utils.Identity()
        elapsed = time.time() - start
        print("Built pre-trained CNN %s in %d seconds."%(image_model_name, elapsed))

    def load_image_from_path(self, path_img):
        """ Load an image given its full path in disk into a tensor
            ready to be used in a pretrained CNN.

        Args:
            path_img    The full path to the image file on disk.
        Returns:
                        The pytorch Variable to be used in the pre-trained CNN
                        that corresponds to the image after all pre-processing.
        """
        input_img = self.load_img(path_img)
        input_tensor = self.tf_img(input_img)
        input_var = torch.autograd.Variable(input_tensor.unsqueeze(0), requires_grad=False)
        return input_var

    def get_global_features(self, input):
        """ Returns features before the application of the last linear transformation.
            In the case of a ResNet, it will be a [1, 2048] tensor."""
        return self.model(input)

    def get_local_features(self, input):
        """ Returns features before the application of the first pooling/fully-connected layer.
            In the case of a ResNet, it will be a [1, 2048, 7, 7] tensor."""
        if self.pretrained_cnn.startswith('vgg') or self.pretrained_cnn.startswith('alexnet'):
            feats = self.model._features(input)
        else:
            feats = self.model.features(input)
        return feats

    def get_all_features(self, input, grain=None):
        if self.pretrained_cnn.startswith('vgg'):
            return self.get_vgg19_features(input, grain=grain)
        elif self.pretrained_cnn.startswith('resnet'):
            return self.get_resnet50_features(input, grain=grain)
        else:
            return self.get_alexnet_features(input, grain=grain)

    def get_alexnet_features(self, input, grain='model'):  # no fc-1000 and softmax, dont' save the maxpooling vector
        if grain == 'model':  # model-grain
            tmp = self.model._features[1](self.model._features[0](input))
            tmp = self.model._features[2](tmp)

            tmp = self.model._features[4](self.model._features[3](tmp))
            tmp = self.model._features[5](tmp)

            tmp = self.model._features[7](self.model._features[6](tmp))
            tmp = self.model._features[9](self.model._features[8](tmp))
            tmp = self.model._features[11](self.model._features[10](tmp))
            x8 = self.model._features[12](tmp)

            tmp = x8.view(x8.size(0), 256 * 6 * 6)
            tmp = self.model.dropout0(tmp)
            tmp = self.model.linear0(tmp)
            tmp = self.model.relu0(tmp)

            x10 = self.model.dropout1(tmp)
            x10 = self.model.linear1(x10)
            x10 = self.model.relu1(x10)
            x10 = self.model.last_linear(x10)  # last_linear == Indentity(forward=x)

            return (input, x8, None, x10)
        elif grain == 'layer':  # layer-grain
            x1 = self.model._features[1](self.model._features[0](input))
            x2 = self.model._features[2](x1)

            x3 = self.model._features[4](self.model._features[3](x2))
            x4 = self.model._features[5](x3)

            x5 = self.model._features[7](self.model._features[6](x4))
            x6 = self.model._features[9](self.model._features[8](x5))
            x7 = self.model._features[11](self.model._features[10](x6))
            x8 = self.model._features[12](x7)

            x9 = x8.view(x8.size(0), 256 * 6 * 6)
            x9 = self.model.dropout0(x9)
            x9 = self.model.linear0(x9)
            x9 = self.model.relu0(x9)

            x10 = self.model.dropout1(x9)
            x10 = self.model.linear1(x10)
            x10 = self.model.relu1(x10)
            x10 = self.model.last_linear(x10)  # last_linear == Indentity(forward=x)

            return (input, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)

    def get_vgg19_features(self, input, grain='model'):  # no fc-1000 and softmax, dont' save the maxpooling vector
        if grain == 'model':  # model-grain
            tmp = self.model._features[1](self.model._features[0](input))
            tmp = self.model._features[3](self.model._features[2](tmp))
            tmp = self.model._features[4](tmp)

            tmp = self.model._features[6](self.model._features[5](tmp))
            tmp = self.model._features[8](self.model._features[7](tmp))
            tmp = self.model._features[9](tmp)

            tmp = self.model._features[11](self.model._features[10](tmp))
            tmp = self.model._features[13](self.model._features[12](tmp))
            tmp = self.model._features[15](self.model._features[14](tmp))
            tmp = self.model._features[17](self.model._features[16](tmp))
            tmp = self.model._features[18](tmp)

            tmp = self.model._features[20](self.model._features[19](tmp))
            tmp = self.model._features[22](self.model._features[21](tmp))
            tmp = self.model._features[24](self.model._features[23](tmp))
            tmp = self.model._features[26](self.model._features[25](tmp))
            tmp = self.model._features[27](tmp)

            tmp = self.model._features[29](self.model._features[28](tmp))
            tmp = self.model._features[31](self.model._features[30](tmp))
            tmp = self.model._features[33](self.model._features[32](tmp))
            tmp = self.model._features[35](self.model._features[34](tmp))
            x21 = self.model._features[36](tmp)

            tmp = x21.view(x21.size(0), -1)
            tmp = self.model.linear0(tmp)
            tmp = self.model.relu0(tmp)
            tmp = self.model.dropout0(tmp)

            x23 = self.model.linear1(tmp)
            x23 = self.model.relu1(x23)
            x23 = self.model.dropout1(x23)
            x23 = self.model.last_linear(x23)  # last_linear == Indentity(forward=x)

            return (input, x21, None, x23)
        elif grain == 'block':  # block-grain
            tmp = self.model._features[1](self.model._features[0](input))
            x2 = self.model._features[3](self.model._features[2](tmp))
            x3 = self.model._features[4](x2)

            tmp = self.model._features[6](self.model._features[5](x3))
            x5 = self.model._features[8](self.model._features[7](tmp))
            x6 = self.model._features[9](x5)

            tmp = self.model._features[11](self.model._features[10](x6))
            tmp = self.model._features[13](self.model._features[12](tmp))
            tmp = self.model._features[15](self.model._features[14](tmp))
            x10 = self.model._features[17](self.model._features[16](tmp))
            x11 = self.model._features[18](x10)

            tmp = self.model._features[20](self.model._features[19](x11))
            tmp = self.model._features[22](self.model._features[21](tmp))
            tmp = self.model._features[24](self.model._features[23](tmp))
            x15 = self.model._features[26](self.model._features[25](tmp))
            x16 = self.model._features[27](x15)

            tmp = self.model._features[29](self.model._features[28](x16))
            tmp = self.model._features[31](self.model._features[30](tmp))
            tmp = self.model._features[33](self.model._features[32](tmp))
            x20 = self.model._features[35](self.model._features[34](tmp))
            x21 = self.model._features[36](x20)

            x22 = x21.view(x21.size(0), -1)
            x22 = self.model.linear0(x22)
            x22 = self.model.relu0(x22)
            x22 = self.model.dropout0(x22)

            x23 = self.model.linear1(x22)
            x23 = self.model.relu1(x23)
            x23 = self.model.dropout1(x23)
            x23 = self.model.last_linear(x23)  # last_linear == Indentity(forward=x)

            return (input, x2, x3, x5, x6, x10, x11, x15, x16, x20, x21, x22, x23)
        elif grain == 'layer':  # layer-grain
            x1 = self.model._features[1](self.model._features[0](input))
            x2 = self.model._features[3](self.model._features[2](x1))
            x3 = self.model._features[4](x2)

            x4 = self.model._features[6](self.model._features[5](x3))
            x5 = self.model._features[8](self.model._features[7](x4))
            x6 = self.model._features[9](x5)

            x7 = self.model._features[11](self.model._features[10](x6))
            x8 = self.model._features[13](self.model._features[12](x7))
            x9 = self.model._features[15](self.model._features[14](x8))
            x10 = self.model._features[17](self.model._features[16](x9))
            x11 = self.model._features[18](x10)

            x12 = self.model._features[20](self.model._features[19](x11))
            x13 = self.model._features[22](self.model._features[21](x12))
            x14 = self.model._features[24](self.model._features[23](x13))
            x15 = self.model._features[26](self.model._features[25](x14))
            x16 = self.model._features[27](x15)

            x17 = self.model._features[29](self.model._features[28](x16))
            x18 = self.model._features[31](self.model._features[30](x17))
            x19 = self.model._features[33](self.model._features[32](x18))
            x20 = self.model._features[35](self.model._features[34](x19))
            x21 = self.model._features[36](x20)

            x22 = x21.view(x21.size(0), -1)
            x22 = self.model.linear0(x22)
            x22 = self.model.relu0(x22)
            x22 = self.model.dropout0(x22)

            x23 = self.model.linear1(x22)
            x23 = self.model.relu1(x23)
            x23 = self.model.dropout1(x23)
            x23 = self.model.last_linear(x23)  # last_linear == Indentity(forward=x)

            return (input, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12,
                    x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23)

    def get_resnet50_features(self, input, grain='model'):  # no fc and softmax, extract features before pooling
        if grain == 'gan':  # model-grain
            tmp = self.model.conv1(input)
            tmp = self.model.bn1(tmp)
            tmp = self.model.relu(tmp)
            tmp = self.model.maxpool(tmp)

            tmp = self.model.layer1(tmp)
            tmp = self.model.layer2(tmp)
            tmp = self.model.layer3(tmp)
            tmp = self.model.layer4(tmp)

            x7 = self.model.avgpool(tmp)
            x7 = x7.view(x7.size(0), -1)
            x7 = self.model.last_linear(x7)  # last_linear == Indentity(forward=x)

            return (input, x7)
        elif grain == 'model':  # model-grain
            tmp = self.model.conv1(input)
            tmp = self.model.bn1(tmp)
            tmp = self.model.relu(tmp)
            tmp = self.model.maxpool(tmp)

            tmp = self.model.layer1(tmp)
            tmp = self.model.layer2(tmp)
            tmp = self.model.layer3(tmp)
            tmp = self.model.layer4(tmp)

            x7 = self.model.avgpool(tmp)
            x7 = x7.view(x7.size(0), -1)
            x7 = self.model.last_linear(x7)  # last_linear == Indentity(forward=x)

            return (input, tmp, x7)
        elif grain == 'block':
            x1 = self.model.conv1(input)
            x1 = self.model.bn1(x1)
            x1 = self.model.relu(x1)
            x2 = self.model.maxpool(x1)

            x3 = self.model.layer1(x2)
            x4 = self.model.layer2(x3)
            x5 = self.model.layer3(x4)
            x6 = self.model.layer4(x5)

            x7 = self.model.avgpool(x6)
            x7 = x7.view(x7.size(0), -1)
            x7 = self.model.last_linear(x7)  # last_linear == Indentity(forward=x)
            return (input, x1, x2, x3, x4, x5, x6, x7)
        elif grain == 'layer':
            def blocks(layers, x):
                outs = tuple()
                for i, layer in enumerate(layers):
                    identity = x

                    out = layer.conv1(x)
                    out = layer.bn1(out)
                    out = layer.relu(out)

                    out1 = layer.conv2(out)
                    out1 = layer.bn2(out1)
                    out1 = layer.relu(out1)

                    out2 = layer.conv3(out1)
                    out2 = layer.bn3(out2)

                    if layer.downsample is not None:
                        identity = layer.downsample(x)

                    out2 += identity
                    out2 = layer.relu(out2)
                    x = out2

                    outs += (out, out1, out2)
                return outs

            x1 = self.model.conv1(input)
            x1 = self.model.bn1(x1)
            x1 = self.model.relu(x1)
            x2 = self.model.maxpool(x1)

            x3 = blocks(self.model.layer1, x2)
            x4 = blocks(self.model.layer2, x3[-1]) # the bottom tensor of the last sublayer
            x5 = blocks(self.model.layer3, x4[-1])
            x6 = blocks(self.model.layer4, x5[-1])

            x7 = self.model.avgpool(x6[-1])
            x7 = x7.view(x7.size(0), -1)
            x7 = self.model.last_linear(x7)  # last_linear == Indentity(forward=x)
            return (input, x1, x2,) + x3 + x4 + x5 + x6 + (x7,)


class PretrainedGenCNN(object):
    """
    Class that encompasses loading of pre-trained CNN models.
    """
    def __init__(self, pretrained_cnn, embed_dim):  # <fix>, assign the fc of gen_cnn manually
        self.pretrained_cnn = pretrained_cnn
        self.build_load_pretrained_cnn(embed_dim)  # <fix>, assign the fc of gen_cnn manually

    # <fix>, general in all cnn variant
    def load_pretrained(self, model, num_classes, settings):
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        # state_dict = model_zoo.load_url(settings['url'])
        # state_dict = update_state_dict(state_dict)
        # model.load_state_dict(state_dict)
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
        return model

    def resnet50(self, num_classes=1000, pretrained='imagenet', embed_dim=None): # <fix>, assign the fc of gen_cnn manually
        """Constructs a ResNet-50 model.
        """
        import myresnet
        model = myresnet.resnet50(pretrained=False, embed_dim=embed_dim) # <fix>, assign the fc of gen_cnn manually
        if pretrained is not None:
            settings = pretrained_settings['resnet50'][pretrained]
            model = self.load_pretrained(model, num_classes, settings)
        # model = self.modify_resnets(model) # Modify attributs has been completed in myresnet
        return model

    def vgg19(self, num_classes=1000, pretrained='imagenet', embed_dim=None): # <fix>, assign the fc of gen_cnn manually
        """VGG 19-layer model (configuration "E")
        """
        import myvgg
        model = myvgg.vgg19(pretrained=False, embed_dim=embed_dim)
        if pretrained is not None:
            settings = pretrained_settings['vgg19'][pretrained]
            model = self.load_pretrained(model, num_classes, settings)
        # model = self.modify_vggs(model) # Modify attributs has been completed in myvgg
        return model

    def alexnet(self, num_classes=1000, pretrained='imagenet', embed_dim=None): # <fix>, assign the fc of gen_cnn manually
        r"""AlexNet model architecture from the
        `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
        """
        # https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
        import myalexnet
        model = myalexnet.alexnet(pretrained=False, embed_dim=embed_dim)
        if pretrained is not None:
            settings = pretrained_settings['alexnet'][pretrained]
            model = self.load_pretrained(model, num_classes, settings)
        # model = self.modify_alexnet(model) # Modify attributs has been completed in myvgg
        return model

    def build_load_pretrained_cnn(self, embed_dim):  # <fix>, assign the fc of gen_cnn manually
        """
            Load a pre-trained CNN using torchvision/cadene.
            Set it into feature extraction mode.
        """
        start = time.time()
        self.load_img = pretrainedmodels.utils.LoadImage()
        image_model_name = self.pretrained_cnn
        self.model = getattr(self, image_model_name)(num_classes=1000, pretrained='imagenet', embed_dim=embed_dim) # <fix>, assign the fc of gen_cnn manually
        # self.model = pretrainedmodels.__dict__[image_model_name](num_classes=1000, pretrained='imagenet')
        #self.model.train()
        self.tf_img = pretrainedmodels.utils.TransformImage(self.model)
        # returns features before the application of the last linear transformation
        # in the case of a resnet152, it will be a [1, 2048] tensor
        # self.model.last_linear = pretrainedmodels.utils.Identity()
        elapsed = time.time() - start
        print("Built pre-trained CNN %s in %d seconds."%(image_model_name, elapsed))

    def get_all_features(self, input, grain=None):
        if self.pretrained_cnn.startswith('vgg'):
            return self.get_vgg19_features(input, grain=grain)
        elif self.pretrained_cnn.startswith('resnet'):
            return self.get_resnet50_features(input, grain=grain)
        else:
            return self.get_alexnet_features(input, grain=grain)

    def get_alexnet_features(self, input, grain='model'):  # no fc-1000 and softmax, dont' save the maxpooling vector
        if grain == 'model':  # model-grain
            tmp = self.model.last_linear(input)
            tmp = self.model.dropout1(tmp)
            tmp = self.model.linear1(tmp)
            tmp = self.model.relu1(tmp)

            x3 = self.model.dropout0(tmp)
            x3 = self.model.linear0(x3)
            x3 = self.model.relu0(x3)
            x3 = x3.view(x3.size(0), 256, 6, 6)

            tmp = self.model._features[11](self.model._features[12](x3))
            tmp = self.model._features[11](self.model._features[10](tmp))
            tmp = self.model._features[9](self.model._features[8](tmp))
            tmp = self.model._features[7](self.model._features[6](tmp))

            tmp = self.model._features[4](self.model._features[5](tmp))
            tmp = self.model._features[4](self.model._features[3](tmp))

            tmp = self.model._features[1](self.model._features[2](tmp))
            x11 = self.model._features[0](tmp)

            return (x3, x11)
        elif grain == 'layer':  # layer-grain
            x1 = self.model.last_linear(input)
            x2 = self.model.dropout1(x1)
            x2 = self.model.linear1(x2)
            x2 = self.model.relu1(x2)

            x3 = self.model.dropout0(x2)
            x3 = self.model.linear0(x3)
            x3 = self.model.relu0(x3)
            x3 = x3.view(x3.size(0), 256, 6, 6)

            x4 = self.model._features[11](self.model._features[12](x3))
            x5 = self.model._features[11](self.model._features[10](x4))
            x6 = self.model._features[9](self.model._features[8](x5))
            x7 = self.model._features[7](self.model._features[6](x6))

            x8 = self.model._features[4](self.model._features[5](x7))
            x9 = self.model._features[4](self.model._features[3](x8))

            x10 = self.model._features[1](self.model._features[2](x9))
            x11 = self.model._features[0](x10)

            return (x3, x4, x5, x6, x7, x8, x9, x10, x11)

    def get_vgg19_features(self, input, grain='model'):  # no fc-1000 and softmax, dont' save the maxpooling vector
        if grain == 'model':   # model-grain
            tmp = self.model.last_linear(input)
            tmp = self.model.dropout1(tmp)

            tmp = self.model.linear1(tmp)
            tmp = self.model.relu1(tmp)
            tmp = self.model.dropout0(tmp)

            x3 = self.model.linear0(tmp)
            x3 = self.model.relu0(x3)
            x3 = x3.view(x3.size(0), 512, 7, 7)

            tmp = self.model._features[35](self.model._features[36](x3))  # add relu after deconv
            tmp = self.model._features[35](self.model._features[34](tmp))
            tmp = self.model._features[33](self.model._features[32](tmp))
            tmp = self.model._features[31](self.model._features[30](tmp))
            tmp = self.model._features[29](self.model._features[28](tmp))

            tmp = self.model._features[26](self.model._features[27](tmp))  # add relu after deconv
            tmp = self.model._features[26](self.model._features[25](tmp))
            tmp = self.model._features[24](self.model._features[23](tmp))
            tmp = self.model._features[22](self.model._features[21](tmp))
            tmp = self.model._features[20](self.model._features[19](tmp))

            tmp = self.model._features[17](self.model._features[18](tmp))  # add relu after deconv
            tmp = self.model._features[17](self.model._features[16](tmp))
            tmp = self.model._features[15](self.model._features[14](tmp))
            tmp = self.model._features[13](self.model._features[12](tmp))
            tmp = self.model._features[11](self.model._features[10](tmp))

            tmp = self.model._features[8](self.model._features[9](tmp))  # add relu after deconv
            tmp = self.model._features[8](self.model._features[7](tmp))
            tmp = self.model._features[6](self.model._features[5](tmp))

            tmp = self.model._features[3](self.model._features[4](tmp))  # add relu after deconv
            tmp = self.model._features[3](self.model._features[2](tmp))
            x24 = self.model._features[0](tmp)  # remove relu before synthesis image vector

            return (x3, x24)
        elif grain == 'block':  # block-grain
            tmp = self.model.last_linear(input)
            tmp = self.model.dropout1(tmp)

            tmp = self.model.linear1(tmp)
            tmp = self.model.relu1(tmp)
            tmp = self.model.dropout0(tmp)

            x3 = self.model.linear0(tmp)
            x3 = self.model.relu0(x3)
            x3 = x3.view(x3.size(0), 512, 7, 7)

            x4 = self.model._features[35](self.model._features[36](x3))  # add relu after deconv
            tmp = self.model._features[35](self.model._features[34](x4))
            tmp = self.model._features[33](self.model._features[32](tmp))
            tmp = self.model._features[31](self.model._features[30](tmp))
            x8 = self.model._features[29](self.model._features[28](tmp))

            x9 = self.model._features[26](self.model._features[27](x8))  # add relu after deconv
            tmp = self.model._features[26](self.model._features[25](x9))
            tmp = self.model._features[24](self.model._features[23](tmp))
            tmp = self.model._features[22](self.model._features[21](tmp))
            x13 = self.model._features[20](self.model._features[19](tmp))

            x14 = self.model._features[17](self.model._features[18](x13))  # add relu after deconv
            tmp = self.model._features[17](self.model._features[16](x14))
            tmp = self.model._features[15](self.model._features[14](tmp))
            tmp = self.model._features[13](self.model._features[12](tmp))
            x18 = self.model._features[11](self.model._features[10](tmp))

            x19 = self.model._features[8](self.model._features[9](x18))  # add relu after deconv
            tmp = self.model._features[8](self.model._features[7](x19))
            x21 = self.model._features[6](self.model._features[5](tmp))

            x22 = self.model._features[3](self.model._features[4](x21))  # add relu after deconv
            tmp = self.model._features[3](self.model._features[2](x22))
            x24 = self.model._features[0](tmp)  # remove relu before synthesis image vector

            return (x3, x4, x8, x9, x13, x14, x18, x19, x21, x22, x24)
        elif grain == 'layer':  # layer-grain
            x1 = self.model.last_linear(input)
            x1 = self.model.dropout1(x1)

            x2 = self.model.linear1(x1)
            x2 = self.model.relu1(x2)
            x2 = self.model.dropout0(x2)

            x3 = self.model.linear0(x2)
            x3 = self.model.relu0(x3)
            x3 = x3.view(x3.size(0), 512, 7, 7)

            x4 = self.model._features[35](self.model._features[36](x3)) # add relu after deconv
            x5 = self.model._features[35](self.model._features[34](x4))
            x6 = self.model._features[33](self.model._features[32](x5))
            x7 = self.model._features[31](self.model._features[30](x6))
            x8 = self.model._features[29](self.model._features[28](x7))

            x9 = self.model._features[26](self.model._features[27](x8)) # add relu after deconv
            x10 = self.model._features[26](self.model._features[25](x9))
            x11 = self.model._features[24](self.model._features[23](x10))
            x12 = self.model._features[22](self.model._features[21](x11))
            x13 = self.model._features[20](self.model._features[19](x12))

            x14 = self.model._features[17](self.model._features[18](x13)) # add relu after deconv
            x15 = self.model._features[17](self.model._features[16](x14))
            x16 = self.model._features[15](self.model._features[14](x15))
            x17 = self.model._features[13](self.model._features[12](x16))
            x18 = self.model._features[11](self.model._features[10](x17))

            x19 = self.model._features[8](self.model._features[9](x18)) # add relu after deconv
            x20 = self.model._features[8](self.model._features[7](x19))
            x21 = self.model._features[6](self.model._features[5](x20))

            x22 = self.model._features[3](self.model._features[4](x21)) # add relu after deconv
            x23 = self.model._features[3](self.model._features[2](x22))
            x24 = self.model._features[0](x23)  # remove relu before synthesis image vector

            return (x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14,
                    x15, x16, x17, x18, x19, x20, x21, x22, x23, x24)

    def get_resnet50_features(self, input, grain='model'):  # no fc and softmax, extract features before pooling
        if grain == 'gan':
            def blocks(blocks, x):
                out = None
                for i, block in enumerate(blocks[::-1]):  # <fix>, reverse order
                    identity = x

                    out = block.conv3(x)
                    out = block.bn3(out)
                    out = block.relu(out)

                    out = block.conv2(out)
                    out = block.bn2(out)
                    out = block.relu(out)

                    out = block.conv1(out)
                    out = block.bn1(out)

                    if block.unsample is not None:
                        identity = block.unsample(x)

                    out += identity
                    out = block.relu(out)
                return out

            tmp = self.model.fc(input)
            x1 = tmp.unsqueeze(-1).unsqueeze(-1)  # x1 = x1[:, :, None, None]
            x1 = self.model.avgunpool(x1)
            x1 = self.model.avgunpool_bn(x1)
            x1 = self.model.avgunpool_relu(x1)

            tmp = blocks(self.model.layer4, x1)  # the top tensor of the first sublayer
            tmp = blocks(self.model.layer3, tmp)
            tmp = blocks(self.model.layer2, tmp)
            tmp = blocks(self.model.layer1, tmp)

            tmp = self.model.maxunpool(tmp)
            tmp = self.model.bn1(tmp)
            tmp = self.model.relu(tmp)
            x7 = self.model.deconv1(tmp)

            return (x1, x7)
        elif grain == 'model':
            def blocks(blocks, x):
                out = None
                for i, block in enumerate(blocks[::-1]):  # <fix>, reverse order
                    identity = x

                    out = block.conv3(x)
                    out = block.bn3(out)
                    out = block.relu(out)

                    out = block.conv2(out)
                    out = block.bn2(out)
                    out = block.relu(out)

                    out = block.conv1(out)
                    out = block.bn1(out)

                    if block.unsample is not None:
                        identity = block.unsample(x)

                    out += identity
                    out = block.relu(out)
                return out

            tmp = self.model.fc(input)
            x1 = tmp.unsqueeze(-1).unsqueeze(-1)  # x1 = x1[:, :, None, None]
            x1 = self.model.avgunpool(x1)
            x1 = self.model.avgunpool_bn(x1)
            x1 = self.model.avgunpool_relu(x1)

            tmp = blocks(self.model.layer4, x1)  # the top tensor of the first sublayer
            tmp = blocks(self.model.layer3, tmp)
            tmp = blocks(self.model.layer2, tmp)
            tmp = blocks(self.model.layer1, tmp)

            tmp = self.model.maxunpool(tmp)
            tmp = self.model.bn1(tmp)
            tmp = self.model.relu(tmp)
            x7 = self.model.deconv1(tmp)

            return (x1, x7)
        elif grain == 'block':
            def blocks(blocks, x):
                out = None
                for i, block in enumerate(blocks[::-1]):  # <fix>, reverse order
                    identity = x

                    out = block.conv3(x)
                    out = block.bn3(out)
                    out = block.relu(out)

                    out = block.conv2(out)
                    out = block.bn2(out)
                    out = block.relu(out)

                    out = block.conv1(out)
                    out = block.bn1(out)

                    if block.unsample is not None:
                        identity = block.unsample(x)

                    out += identity
                    out = block.relu(out)
                return out

            tmp = self.model.fc(input)
            x1 = tmp.unsqueeze(-1).unsqueeze(-1)  # x1 = x1[:, :, None, None]
            x1 = self.model.avgunpool(x1)
            x1 = self.model.avgunpool_bn(x1)
            x1 = self.model.avgunpool_relu(x1)

            x2 = blocks(self.model.layer4, x1)  # the top tensor of the first sublayer
            x3 = blocks(self.model.layer3, x2)
            x4 = blocks(self.model.layer2, x3)
            x5 = blocks(self.model.layer1, x4)

            x6 = self.model.maxunpool(x5)
            x6 = self.model.bn1(x6)
            x6 = self.model.relu(x6)
            x7 = self.model.deconv1(x6)

            return (x1, x2, x3, x4, x5, x6, x7)
        elif grain == 'layer':
            def blocks(blocks, x):
                outs = tuple()
                for i, block in enumerate(blocks[::-1]):  # <fix>, reverse order
                    identity = x

                    out = block.conv3(x)
                    out = block.bn3(out)
                    out = block.relu(out)

                    out1 = block.conv2(out)
                    out1 = block.bn2(out1)
                    out1 = block.relu(out1)

                    out2 = block.conv1(out1)
                    out2 = block.bn1(out2)

                    if block.unsample is not None:
                        identity = block.unsample(x)

                    out2 += identity
                    out2 = block.relu(out2)
                    x = out2

                    outs += (out, out1, out2)
                return outs

            tmp = self.model.fc(input)
            x1 = tmp.unsqueeze(-1).unsqueeze(-1)  # x1 = x1[:, :, None, None]
            x1 = self.model.avgunpool(x1)
            x1 = self.model.avgunpool_bn(x1)
            x1 = self.model.avgunpool_relu(x1)

            x2 = blocks(self.model.layer4, x1)  # the top tensor of the first sublayer
            x3 = blocks(self.model.layer3, x2[-1])
            x4 = blocks(self.model.layer2, x3[-1])
            x5 = blocks(self.model.layer1, x4[-1])

            x6 = self.model.maxunpool(x5[-1])
            x6 = self.model.bn1(x6)
            x6 = self.model.relu(x6)
            x7 = self.model.deconv1(x6)

            return (x1,) + x2 + x3 + x4 + x5 + (x6, x7)
