from typing import Tuple


def get_rectangle_area(width : int, height : int) -> int:
    return width * height


def get_intersection(
    x_top_left_1: int,
    y_top_left_1: int,
    x_bottom_right_1: int,
    y_bottom_right_1: int,
    x_top_left_2: int,
    y_top_left_2: int,
    x_bottom_right_2: int,
    y_bottom_right_2: int,
) -> Tuple[int, int, int, int]:
    x_top_left = max(x_top_left_1, x_top_left_2)
    y_top_left = max(y_top_left_1, y_top_left_2)
    x_bottom_right = min(x_bottom_right_1, x_bottom_right_2)
    y_bottom_right = min(y_bottom_right_1, y_bottom_right_2)
    return x_top_left, y_top_left, x_bottom_right, y_bottom_right


def get_intersection_area(
    x_top_left_1: int,
    y_top_left_1: int,
    x_bottom_right_1: int,
    y_bottom_right_1: int,
    x_top_left_2: int,
    y_top_left_2: int,
    x_bottom_right_2: int,
    y_bottom_right_2: int,
) -> int:
    x_top_left, y_top_left, x_bottom_right, y_bottom_right = get_intersection(
        x_top_left_1=x_top_left_1,
        y_top_left_1=y_top_left_1,
        x_bottom_right_1=x_bottom_right_1,
        y_bottom_right_1=y_bottom_right_1,
        x_top_left_2=x_top_left_2,
        y_top_left_2=y_top_left_2,
        x_bottom_right_2=x_bottom_right_2,
        y_bottom_right_2=y_bottom_right_2,
    )
    if x_top_left > x_bottom_right or y_top_left > y_bottom_right:
        # no intersection
        return 0
    width = x_bottom_right - x_top_left
    height = y_bottom_right - y_top_left
    return get_rectangle_area(width=width, height=height)


def get_top_left_bottom_right_coordinates(
    x_center, y_center, width, height
) -> Tuple[int, int, int, int]:
    x_top_left = x_center - width // 2
    y_top_left = y_center - height // 2
    x_bottom_right = x_center + width // 2
    y_bottom_right = y_center + height // 2
    return x_top_left, y_top_left, x_bottom_right, y_bottom_right


def IoU(
    x_center_1: int,
    y_center_1: int,
    w_1: int,
    h_1: int,
    x_center_2: int,
    y_center_2: int,
    w_2: int,
    h_2: int,
) -> float:
    area_rectangle_1 = get_rectangle_area(width=w_1, height=h_1)
    area_rectangle_2 = get_rectangle_area(width=w_2, height=h_2)
    top_left_bottom_right_coordinates_rectangle_1 = (
        get_top_left_bottom_right_coordinates(
            x_center=x_center_1, y_center=y_center_1, width=w_1, height=h_1
        )
    )
    top_left_bottom_right_coordinates_rectangle_2 = (
        get_top_left_bottom_right_coordinates(
            x_center=x_center_2, y_center=y_center_2, width=w_2, height=h_2
        )
    )
    area_intersection = get_intersection_area(
        *top_left_bottom_right_coordinates_rectangle_1,
        *top_left_bottom_right_coordinates_rectangle_2
    )
    area_union = area_rectangle_1 + area_rectangle_2 - area_intersection
    # Prevent division by zero in case of degenerate rectangles 
    return area_intersection / area_union if area_union > 0 else 0.0
