import torch


class YoloModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
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
        self.activation_1 = torch.nn.LeakyReLU()
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        return torch.nn.Sequential(self.conv_1, self.activation_1, self.max_pool_1)

    def get_block_2(self) -> torch.nn.Sequential:
        self.conv_2 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=192,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
        )
        self.activation_2 = torch.nn.LeakyReLU()
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        return torch.nn.Sequential(self.conv_2, self.activation_2, self.max_pool_2)

    def get_block_3(self) -> torch.nn.Sequential:
        self.conv_3 = torch.nn.Conv2d(
            in_channels=192,
            out_channels=128,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
        )
        self.activation_3 = torch.nn.LeakyReLU()

        self.conv_4 = torch.nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
        )
        self.activation_4 = torch.nn.LeakyReLU()

        self.conv_5 = torch.nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
        )
        self.activation_5 = torch.nn.LeakyReLU()

        self.conv_6 = torch.nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
        )
        self.activation_6 = torch.nn.LeakyReLU()

        self.max_pool_3 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        return torch.nn.Sequential(
            self.conv_3,
            self.activation_3,
            self.conv_4,
            self.activation_4,
            self.conv_5,
            self.activation_5,
            self.conv_6,
            self.activation_6,
            self.max_pool_3,
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
            self.sub_block.append(torch.nn.LeakyReLU())

        self.conv_15 = torch.nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
        )
        self.activation_15 = torch.nn.LeakyReLU()

        self.conv_16 = torch.nn.Conv2d(
            in_channels=512,
            out_channels=1024,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
        )
        self.activation_16 = torch.nn.LeakyReLU()

        self.max_pool_4 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        return torch.nn.Sequential(
            *self.sub_block,
            self.conv_15,
            self.activation_15,
            self.conv_16,
            self.activation_16,
            self.max_pool_4,
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
            self.sub_block_2.append(torch.nn.LeakyReLU())

        self.conv_21 = torch.nn.Conv2d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
        )
        self.activation_21 = torch.nn.LeakyReLU()

        self.conv_22 = torch.nn.Conv2d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            dilation=(1, 1),
        )
        self.activation_22 = torch.nn.LeakyReLU()

        return torch.nn.Sequential(
            *self.sub_block_2,
            self.conv_21,
            self.activation_21,
            self.conv_22,
            self.activation_22,
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
        self.activation_23 = torch.nn.LeakyReLU()

        self.conv_24 = torch.nn.Conv2d(
            in_channels=1024,
            out_channels=1024,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
        )
        self.activation_24 = torch.nn.LeakyReLU()

        return torch.nn.Sequential(
            self.conv_23, self.activation_23, self.conv_24, self.activation_24
        )

    def get_linear_block_1(self) -> torch.nn.Linear:
        self.linear_1 = torch.nn.Linear(50_176, 4_096)
        self.activation_25 = torch.nn.LeakyReLU()
        return torch.nn.Sequential(self.linear_1, self.activation_25)

    def get_linear_block_2(self) -> torch.nn.Linear:
        self.linear_2 = torch.nn.Linear(4_096, 7 * 7 * 30)
        return torch.nn.Sequential(self.linear_2)

    def forward(self, x) -> torch.tensor:
        for block in self.blocks:
            x = block(x)
            print(x.shape)

        x = x.reshape(-1, 50_176)
        print(x.shape)

        x = self.linear_block_1(x)
        print(x.shape)

        x = self.linear_block_2(x)
        print(x.shape)

        y = x.reshape(-1, 7, 7, 30)
        print(y.shape)

        return y
