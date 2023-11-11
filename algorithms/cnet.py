import torch
import torch.nn as nn

class CNET(nn.Module):
    def __init__(self,
                 residual_channels=64,
                 filter_width=3,
                 dilations=[1, 2, 4, 8, 1, 2, 4, 8],
                 input_channels=123,
                 output_channels=48,
                 cond_dim=None,
                 cond_channels=64,
                 postnet_channels=256,
                 do_postproc=True,
                 do_GU=True):

        super(CNET, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.filter_width = filter_width
        self.dilations = dilations
        self.residual_channels = residual_channels
        self.postnet_channels = postnet_channels
        self.do_postproc = do_postproc
        self.do_GU = do_GU

        if cond_dim is not None:
            self._use_cond = True
            self.cond_dim = cond_dim
            self.cond_channels = cond_channels
        else:
            self._use_cond = False

        self._create_variables()

    def _create_variables(self):
        fw = self.filter_width
        r = self.residual_channels
        s = self.postnet_channels

        self.input_layer = nn.Conv1d(self.input_channels, 2*r, fw, padding=fw//2)

        if self._use_cond:
            self.embed_cond = nn.Conv1d(self.cond_dim, self.cond_channels, 1)

        self.conv_modules = nn.ModuleList()
        for i, dilation in enumerate(self.dilations):
            conv_module = nn.ModuleDict({
                'filter_gate': nn.Conv1d(r, 2*r, fw, padding=dilation*(fw//2), dilation=dilation),
            })
            if self.do_postproc:
                conv_module['skip_gate'] = nn.Conv1d(r, s, 1)
            if self.do_GU:
                conv_module['post_filter_gate'] = nn.Conv1d(r, r, 1)
            if self._use_cond:
                conv_module['cond_filter_gate'] = nn.Conv1d(self.cond_channels, 2*r, 1)
            self.conv_modules.append(conv_module)

        if self.do_postproc:
            self.postproc_module = nn.Sequential(
                nn.Conv1d(s, s, fw, padding=fw//2),
                nn.ReLU(),
                nn.Conv1d(s, sum(self.output_channels) if isinstance(self.output_channels, list) else self.output_channels, fw, padding=fw//2)
            )

        self.last_layer = nn.Conv1d(r, sum(self.output_channels) if isinstance(self.output_channels, list) else self.output_channels, fw, padding=fw//2)

    def forward(self, X_input, cond_input=None):
        R = self._input_layer(X_input)
        X = R
        skip_outputs = []
        for i, dilation in enumerate(self.dilations):
            X, skip = self._conv_module(X, R, i, dilation, cond_input)
            skip_outputs.append(skip)

        if self.do_postproc:
            Y = self._postproc_module(skip_outputs)
        else:
            Y = self._last_layer(X)

        return Y

    def _input_layer(self, main_input):
        X = self.input_layer(main_input)
        r = self.residual_channels
        Y = torch.tanh(X[:, :r]) * torch.sigmoid(X[:, r:])
        return Y

    def _embed_cond(self, cond_input):
        Y = self.embed_cond(cond_input)
        return torch.tanh(Y)

    def _conv_module(self, main_input, residual_input, module_idx, dilation, cond_input=None):
        conv_module = self.conv_modules[module_idx]
        X = main_input
        Y = conv_module['filter_gate'](X)
        if self._use_cond:
            C = conv_module['cond_filter_gate'](cond_input)
            C = torch.tanh(C)
            Y += C
        Y = torch.tanh(Y[:, :r]) * torch.sigmoid(Y[:, r:])
        if self.do_postproc:
            skip_out = conv_module['skip_gate'](Y)
        else:
            skip_out = []
        if self.do_GU:
            Y = conv_module['post_filter_gate'](Y)
            Y += X
        return Y, skip_out

    def _postproc_module(self, residual_module_outputs):
        X = sum(residual_module_outputs)
        Y = self.postproc_module(X)
        if isinstance(self.output_channels, list):
            output_list = []
            start = 0
            for channels in self.output_channels:
                output_list.append(Y[:, start:start+channels])
                start += channels
            Y = output_list
        return Y

    def _last_layer(self, last_layer_ip):
        Y = self.last_layer(last_layer_ip)
        if isinstance(self.output_channels, list):
            output_list = []
            start = 0
            for channels in self.output_channels:
                output_list.append(Y[:, start:start+channels])
                start += channels
            Y = output_list
        return Y
