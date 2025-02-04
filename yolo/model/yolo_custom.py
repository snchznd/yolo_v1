import torch


class CustomYoloModel(torch.nn.Module):

    def __init__(self, dropout_probability: float = 0.25):
        super().__init__()
        self.dropout_probability = dropout_probability
        self.blocks = torch.nn.ModuleList()
        self.blocks.append(self.get_block_1())
        self.blocks.append(self.get_block_2())
        self.blocks.append(self.get_block_3())
        self.blocks.append(self.get_block_4())
        self.blocks.append(self.get_block_5())
        self.blocks.append(self.get_block_6())
        self.linear_block_1 = self.get_linear_block_1()
        self.linear_block_2 = self.get_linear_block_2()

    def get_block_1(self) -> torch.nn.Sequential:
        self.conv_1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            dilation=(1, 1),
        )
        self.batch_norm_1 = torch.nn.BatchNorm2d(num_features=64)
        self.activation_1 = torch.nn.LeakyReLU()
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout_1 = torch.nn.Dropout2d(p=self.dropout_probability)
        return torch.nn.Sequential(
            self.conv_1,
            self.batch_norm_1,
            self.activation_1,
            self.max_pool_1,
            self.dropout_1,
        )

    def get_block_2(self) -> torch.nn.Sequential:
        self.conv_2 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=192,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
        )
        self.batch_norm_2 = torch.nn.BatchNorm2d(num_features=192)
        self.activation_2 = torch.nn.LeakyReLU()
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout_2 = torch.nn.Dropout2d(p=self.dropout_probability)
        return torch.nn.Sequential(
            self.conv_2,
            self.batch_norm_2,
            self.activation_2,
            self.max_pool_2,
            self.dropout_2,
        )

    def get_block_3(self) -> torch.nn.Sequential:
        self.conv_3 = torch.nn.Conv2d(
            in_channels=192,
            out_channels=128,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
        )
        self.batch_norm_3 = torch.nn.BatchNorm2d(num_features=128)
        self.activation_3 = torch.nn.LeakyReLU()

        self.conv_4 = torch.nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
        )
        self.batch_norm_4 = torch.nn.BatchNorm2d(num_features=256)
        self.activation_4 = torch.nn.LeakyReLU()

        self.conv_5 = torch.nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
        )
        self.batch_norm_5 = torch.nn.BatchNorm2d(num_features=256)
        self.activation_5 = torch.nn.LeakyReLU()

        self.conv_6 = torch.nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
        )
        self.batch_norm_6 = torch.nn.BatchNorm2d(num_features=512)
        self.activation_6 = torch.nn.LeakyReLU()

        self.max_pool_3 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout_3 = torch.nn.Dropout2d(p=self.dropout_probability)

        return torch.nn.Sequential(
            self.conv_3,
            self.batch_norm_3,
            self.activation_3,
            self.conv_4,
            self.batch_norm_4,
            self.activation_4,
            self.conv_5,
            self.batch_norm_5,
            self.activation_5,
            self.conv_6,
            self.batch_norm_6,
            self.activation_6,
            self.max_pool_3,
            self.dropout_3,
        )

    def get_block_4(self) -> torch.nn.Sequential:
        self.sub_block = torch.nn.ModuleList()
        for x in range(4):
            self.sub_block.append(
                torch.nn.Conv2d(
                    in_channels=512,
                    out_channels=256,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    dilation=(1, 1),
                )
            )
            self.sub_block.append(torch.nn.BatchNorm2d(num_features=256))
            self.sub_block.append(torch.nn.LeakyReLU())
            self.sub_block.append(
                torch.nn.Conv2d(
                    in_channels=256,
                    out_channels=512,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    dilation=(1, 1),
                )
            )
            self.sub_block.append(torch.nn.BatchNorm2d(num_features=512))
            self.sub_block.append(torch.nn.LeakyReLU())

        self.conv_15 = torch.nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
        )
        self.batch_norm_15 = torch.nn.BatchNorm2d(num_features=512)
        self.activation_15 = torch.nn.LeakyReLU()

        self.conv_16 = torch.nn.Conv2d(
            in_channels=512,
            out_channels=1024,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
        )
        self.batch_norm_16 = torch.nn.BatchNorm2d(num_features=1024)
        self.activation_16 = torch.nn.LeakyReLU()

        self.max_pool_4 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout_4 = torch.nn.Dropout2d(p=self.dropout_probability)
        return torch.nn.Sequential(
            *self.sub_block,
            self.conv_15,
            self.batch_norm_15,
            self.activation_15,
            self.conv_16,
            self.batch_norm_16,
            self.activation_16,
            self.max_pool_4,
            self.dropout_4,
        )

    def get_block_5(self) -> torch.nn.Sequential:
        self.sub_block_2 = torch.nn.ModuleList()
        for x in range(2):
            self.sub_block_2.append(
                torch.nn.Conv2d(
                    in_channels=1024,
                    out_channels=512,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    dilation=(1, 1),
                )
            )
            self.sub_block_2.append(torch.nn.BatchNorm2d(num_features=512))
            self.sub_block_2.append(torch.nn.LeakyReLU())
            self.sub_block_2.append(
                torch.nn.Conv2d(
                    in_channels=512,
                    out_channels=1024,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    dilation=(1, 1),
                )
            )
            self.sub_block_2.append(torch.nn.BatchNorm2d(num_features=1024))
            self.sub_block_2.append(torch.nn.LeakyReLU())

        self.conv_21 = torch.nn.Conv2d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
        )
        self.batch_norm_21 = torch.nn.BatchNorm2d(num_features=1024)
        self.activation_21 = torch.nn.LeakyReLU()

        self.conv_22 = torch.nn.Conv2d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            dilation=(1, 1),
        )
        self.batch_norm_22 = torch.nn.BatchNorm2d(num_features=1024)
        self.activation_22 = torch.nn.LeakyReLU()
        self.dropout_5 = torch.nn.Dropout2d(p=self.dropout_probability)

        return torch.nn.Sequential(
            *self.sub_block_2,
            self.conv_21,
            self.batch_norm_21,
            self.activation_21,
            self.conv_22,
            self.batch_norm_22,
            self.activation_22,
            self.dropout_5,
        )

    def get_block_6(self) -> torch.nn.Sequential:
        self.conv_23 = torch.nn.Conv2d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
        )
        self.batch_norm_23 = torch.nn.BatchNorm2d(num_features=1024)
        self.activation_23 = torch.nn.LeakyReLU()

        self.conv_24 = torch.nn.Conv2d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
        )
        self.batch_norm_24 = torch.nn.BatchNorm2d(num_features=1024)
        self.activation_24 = torch.nn.LeakyReLU()
        
        self.dropout_6 = torch.nn.Dropout2d(p=self.dropout_probability)

        return torch.nn.Sequential(
            self.conv_23,
            self.batch_norm_23,
            self.activation_23,
            self.conv_24,
            self.batch_norm_24,
            self.activation_24,
            self.dropout_6
        )

    def get_linear_block_1(self) -> torch.nn.Linear:
        self.linear_1 = torch.nn.Linear(50_176, 4_096)
        self.batch_norm_24 = torch.nn.BatchNorm1d(num_features=4_096)
        self.activation_25 = torch.nn.LeakyReLU()
        self.dropout_7 = torch.nn.Dropout(p=self.dropout_probability)
        return torch.nn.Sequential(self.linear_1, self.batch_norm_24, self.activation_25, self.dropout_7)

    def get_linear_block_2(self) -> torch.nn.Linear:
        self.linear_2 = torch.nn.Linear(4_096, 7 * 7 * 30)
        return torch.nn.Sequential(self.linear_2)

    def forward(self, x) -> torch.tensor:
        for block in self.blocks:
            x = block(x)

        x = x.reshape(-1, 50_176)

        x = self.linear_block_1(x)

        x = self.linear_block_2(x)

        y = x.reshape(-1, 7, 7, 30)

        return y
