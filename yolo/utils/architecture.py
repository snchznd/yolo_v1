from typing import Tuple
from math import floor
from collections import namedtuple

ConvolutionalLayer = namedtuple(
    "ConvolutionalLayer",
    ("input_height", "input_width", "kernel_size", "stride", "padding", "dilation"),
)


def get_conv_layer_output_dim_2d(
    input_height: int,
    input_width: int,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
) -> Tuple[int, int]:
    output_size = []
    input_dimensions = [input_height, input_width]
    for idx in range(2):
        output_size.append(
            get_conv_layer_output_dim_1d(
                input_dimensions[idx],
                kernel_size[idx],
                stride[idx],
                padding[idx],
                dilation[idx],
            )
        )
    return tuple(output_size)


def get_conv_layer_output_dim_1d(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    return floor(
        (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )


def get_identity_padding(
    input_size: int,
    kernel_size: int,
    stride: int,
    dilation: int,
) -> int:
    return int(
        ((stride - 1) * input_size + dilation * (kernel_size - 1) - stride + 1) / 2
    )


if __name__ == "__main__":
    conv_layer_1 = ConvolutionalLayer(
        input_height=448,
        input_width=448,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
        dilation=(1, 1),
    )

    conv_layer_2 = ConvolutionalLayer(
        input_height=14,
        input_width=14,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        dilation=(1, 1),
    )

    output_dim = get_conv_layer_output_dim_2d(
        input_height=conv_layer_2.input_height,
        input_width=conv_layer_2.input_width,
        kernel_size=conv_layer_2.kernel_size,
        stride=conv_layer_2.stride,
        padding=conv_layer_2.padding,
        dilation=conv_layer_2.dilation,
    )
    print(f"input dim:  (h={conv_layer_2.input_height}, w={conv_layer_2.input_width})")
    print(f"output dim: (h={output_dim[0]}, w={output_dim[1]})")
    
    print(get_identity_padding(input_size=28, kernel_size=3, stride=1, dilation=1))
