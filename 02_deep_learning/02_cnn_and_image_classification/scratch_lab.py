from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
FIGURE_PATH = ARTIFACT_DIR / 'cnn_feature_maps.svg'

CLASS_NAMES = ['vertical_bar', 'horizontal_bar']
IMAGE_SIZE = 6
CHANNEL_COUNT = 3
BAR_WIDTH = 2


def round_float(value: float) -> float:
    return round(float(value), 6)


def empty_image() -> list[list[list[float]]]:
    return [
        [[0.0 for _ in range(IMAGE_SIZE)] for _ in range(IMAGE_SIZE)]
        for _ in range(CHANNEL_COUNT)
    ]


def add_vertical_bar(
    image: list[list[list[float]]],
    *,
    col_start: int,
    color: tuple[float, float, float],
) -> None:
    for channel_index, intensity in enumerate(color):
        for row_index in range(IMAGE_SIZE):
            for col_index in range(col_start, col_start + BAR_WIDTH):
                image[channel_index][row_index][col_index] = intensity


def add_horizontal_bar(
    image: list[list[list[float]]],
    *,
    row_start: int,
    color: tuple[float, float, float],
) -> None:
    for channel_index, intensity in enumerate(color):
        for row_index in range(row_start, row_start + BAR_WIDTH):
            for col_index in range(IMAGE_SIZE):
                image[channel_index][row_index][col_index] = intensity


def build_dataset() -> tuple[list[list[list[list[float]]]], list[int]]:
    vertical_left = empty_image()
    add_vertical_bar(vertical_left, col_start=1, color=(1.0, 0.5, 0.2))

    vertical_right = empty_image()
    add_vertical_bar(vertical_right, col_start=3, color=(0.9, 0.45, 0.25))

    horizontal_top = empty_image()
    add_horizontal_bar(horizontal_top, row_start=1, color=(0.2, 1.0, 0.6))

    horizontal_bottom = empty_image()
    add_horizontal_bar(horizontal_bottom, row_start=3, color=(0.25, 0.95, 0.55))

    images = [vertical_left, vertical_right, horizontal_top, horizontal_bottom]
    labels = [0, 0, 1, 1]
    return images, labels


def build_kernels() -> list[list[list[list[float]]]]:
    vertical_base = [
        [-1.0, 2.0, -1.0],
        [-1.0, 2.0, -1.0],
        [-1.0, 2.0, -1.0],
    ]
    horizontal_base = [
        [-1.0, -1.0, -1.0],
        [2.0, 2.0, 2.0],
        [-1.0, -1.0, -1.0],
    ]
    vertical_channel_mix = (1.0, 0.7, 0.4)
    horizontal_channel_mix = (0.4, 1.0, 0.7)

    kernels = []
    for base, mix in (
        (vertical_base, vertical_channel_mix),
        (horizontal_base, horizontal_channel_mix),
    ):
        kernel = []
        for channel_scale in mix:
            kernel.append(
                [
                    [round_float(channel_scale * value) for value in row]
                    for row in base
                ]
            )
        kernels.append(kernel)
    return kernels


def manual_conv_batch(
    images: list[list[list[list[float]]]],
    kernels: list[list[list[list[float]]]],
) -> list[list[list[list[float]]]]:
    kernel_size = len(kernels[0][0])
    output_size = IMAGE_SIZE - kernel_size + 1
    batch_outputs = []
    for image in images:
        feature_maps = []
        for kernel in kernels:
            feature_map = []
            for row_index in range(output_size):
                row = []
                for col_index in range(output_size):
                    total = 0.0
                    for channel_index in range(CHANNEL_COUNT):
                        for kernel_row in range(kernel_size):
                            for kernel_col in range(kernel_size):
                                total += (
                                    image[channel_index][row_index + kernel_row][col_index + kernel_col]
                                    * kernel[channel_index][kernel_row][kernel_col]
                                )
                    row.append(round_float(total))
                feature_map.append(row)
            feature_maps.append(feature_map)
        batch_outputs.append(feature_maps)
    return batch_outputs


def relu_batch(feature_maps: list[list[list[list[float]]]]) -> list[list[list[list[float]]]]:
    activated = []
    for sample in feature_maps:
        sample_maps = []
        for fmap in sample:
            sample_maps.append(
                [[round_float(max(0.0, value)) for value in row] for row in fmap]
            )
        activated.append(sample_maps)
    return activated


def max_pool_batch(
    feature_maps: list[list[list[list[float]]]],
    *,
    pool_size: int = 2,
    stride: int = 2,
) -> list[list[list[list[float]]]]:
    pooled_batch = []
    for sample in feature_maps:
        sample_maps = []
        for fmap in sample:
            pooled = []
            output_size = ((len(fmap) - pool_size) // stride) + 1
            for row_index in range(output_size):
                row = []
                for col_index in range(output_size):
                    values = []
                    start_row = row_index * stride
                    start_col = col_index * stride
                    for delta_row in range(pool_size):
                        for delta_col in range(pool_size):
                            values.append(fmap[start_row + delta_row][start_col + delta_col])
                    row.append(round_float(max(values)))
                pooled.append(row)
            sample_maps.append(pooled)
        pooled_batch.append(sample_maps)
    return pooled_batch


def mean_feature_scores(pooled_maps: list[list[list[list[float]]]]) -> list[list[float]]:
    scores = []
    for sample in pooled_maps:
        sample_scores = []
        for fmap in sample:
            total = sum(sum(row) for row in fmap)
            count = len(fmap) * len(fmap[0])
            sample_scores.append(round_float(total / count))
        scores.append(sample_scores)
    return scores


def argmax_index(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def max_position(feature_map: list[list[float]]) -> dict[str, object]:
    best_row = 0
    best_col = 0
    best_value = feature_map[0][0]
    for row_index, row in enumerate(feature_map):
        for col_index, value in enumerate(row):
            if value > best_value:
                best_row = row_index
                best_col = col_index
                best_value = value
    return {'position': [best_row, best_col], 'value': round_float(best_value)}


def average_channels(image: list[list[list[float]]]) -> list[list[float]]:
    return [
        [
            round_float(sum(image[channel][row][col] for channel in range(CHANNEL_COUNT)) / CHANNEL_COUNT)
            for col in range(IMAGE_SIZE)
        ]
        for row in range(IMAGE_SIZE)
    ]


def flatten_values(grid: list[list[float]]) -> list[float]:
    return [value for row in grid for value in row]


def panel_svg(
    grid: list[list[float]],
    *,
    left: int,
    top: int,
    cell: int,
    title: str,
    subtitle: str,
) -> str:
    flat = flatten_values(grid)
    minimum = min(flat)
    maximum = max(flat)
    span = maximum - minimum if maximum != minimum else 1.0
    nodes = [
        f'<text x="{left}" y="{top - 22}" font-size="16" font-family="Arial, sans-serif">{escape(title)}</text>',
        f'<text x="{left}" y="{top - 6}" font-size="11" font-family="Arial, sans-serif" fill="#555">{escape(subtitle)}</text>',
    ]
    for row_index, row in enumerate(grid):
        for col_index, value in enumerate(row):
            normalized = (value - minimum) / span
            shade = 245 - int(normalized * 170)
            x = left + col_index * cell
            y = top + row_index * cell
            nodes.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="rgb({shade}, {shade}, 255)" stroke="#adb5bd" />'
            )
            nodes.append(
                f'<text x="{x + cell / 2}" y="{y + cell / 2 + 4}" font-size="11" '
                f'font-family="Arial, sans-serif" text-anchor="middle" fill="#1f2937">{value:.2f}</text>'
            )
    return '\n'.join(nodes)


def save_svg(
    images: list[list[list[list[float]]]],
    feature_maps: list[list[list[list[float]]]],
    pooled_maps: list[list[list[list[float]]]],
) -> None:
    cell = 28
    width = 1060
    height = 620
    sample_vertical = 0
    sample_horizontal = 2
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="28" y="34" font-size="28" font-family="Arial, sans-serif">CNN feature maps</text>
  <text x="28" y="58" font-size="13" font-family="Arial, sans-serif" fill="#555">toy RGB-like bars → convolution → max pooling</text>
  {panel_svg(average_channels(images[sample_vertical]), left=28, top=108, cell=cell, title='vertical sample input', subtitle='RGB 평균으로 본 6x6 입력')}
  {panel_svg(feature_maps[sample_vertical][0], left=268, top=108, cell=cell, title='vertical detector feature map', subtitle='세로 줄무늬 detector 반응')}
  {panel_svg(feature_maps[sample_vertical][1], left=548, top=108, cell=cell, title='horizontal detector feature map', subtitle='가로 줄무늬 detector 반응')}
  {panel_svg(pooled_maps[sample_vertical][0], left=828, top=108, cell=cell * 2, title='pooled vertical map', subtitle='4x4 → 2x2 max pooling')}
  {panel_svg(average_channels(images[sample_horizontal]), left=28, top=388, cell=cell, title='horizontal sample input', subtitle='RGB 평균으로 본 6x6 입력')}
  {panel_svg(feature_maps[sample_horizontal][0], left=268, top=388, cell=cell, title='vertical detector on horizontal image', subtitle='세로 detector는 약하게 반응')}
  {panel_svg(feature_maps[sample_horizontal][1], left=548, top=388, cell=cell, title='horizontal detector on horizontal image', subtitle='가로 detector는 강하게 반응')}
  {panel_svg(pooled_maps[sample_horizontal][1], left=828, top=388, cell=cell * 2, title='pooled horizontal map', subtitle='가장 강한 반응만 남는다')}
</svg>
'''
    FIGURE_PATH.write_text(svg, encoding='utf-8')


def run() -> None:
    images, labels = build_dataset()
    kernels = build_kernels()
    raw_feature_maps = manual_conv_batch(images, kernels)
    feature_maps = relu_batch(raw_feature_maps)
    pooled_maps = max_pool_batch(feature_maps)
    scores = mean_feature_scores(pooled_maps)
    predictions = [argmax_index(sample_scores) for sample_scores in scores]
    accuracy = sum(int(pred == gold) for pred, gold in zip(predictions, labels)) / len(labels)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_svg(images=images, feature_maps=feature_maps, pooled_maps=pooled_maps)

    output_size = len(feature_maps[0][0])
    pooled_size = len(pooled_maps[0][0])
    sample_activation_peaks = []
    for sample_index, sample_maps in enumerate(feature_maps):
        sample_activation_peaks.append(
            {
                'sample_index': sample_index,
                'gold_label': CLASS_NAMES[labels[sample_index]],
                'predicted_label': CLASS_NAMES[predictions[sample_index]],
                'vertical_detector': max_position(sample_maps[0]),
                'horizontal_detector': max_position(sample_maps[1]),
            }
        )

    metrics = {
        'dataset_shape': [len(images), CHANNEL_COUNT, IMAGE_SIZE, IMAGE_SIZE],
        'class_names': CLASS_NAMES,
        'labels': [CLASS_NAMES[label] for label in labels],
        'input_channel_count': CHANNEL_COUNT,
        'output_feature_map_count': len(kernels),
        'local_receptive_field': [len(kernels[0][0]), len(kernels[0][0][0])],
        'conv_kernel_shape': [len(kernels), CHANNEL_COUNT, 3, 3],
        'feature_map_shape': [len(images), len(kernels), output_size, output_size],
        'pooled_shape': [len(images), len(kernels), pooled_size, pooled_size],
        'parameter_sharing_reuse_count': output_size * output_size,
        'pooling_reduction_ratio': round_float((output_size * output_size) / (pooled_size * pooled_size)),
        'classification_scores': scores,
        'predictions': [CLASS_NAMES[pred] for pred in predictions],
        'classification_accuracy': round_float(accuracy),
        'sample_activation_peaks': sample_activation_peaks,
        'channel_role_summary': {
            'input_channels': ['red-dominant signal', 'green-dominant signal', 'support color signal'],
            'feature_maps': ['vertical_bar_detector', 'horizontal_bar_detector'],
        },
        'figure_path': 'artifacts/scratch-manual/cnn_feature_maps.svg',
    }
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
