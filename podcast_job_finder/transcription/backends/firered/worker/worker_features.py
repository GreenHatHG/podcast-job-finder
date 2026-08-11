from __future__ import annotations

# Third-party modules are installed in the dedicated FireRed environment.
# pylint: disable=import-error

import math
from pathlib import Path

import kaldi_native_fbank as knf  # type: ignore[import-not-found]
import kaldiio  # type: ignore[import-not-found]
import numpy as np
import soundfile as sf  # type: ignore[import-not-found]


EXPECTED_SAMPLE_RATE = 16_000
MEL_BIN_COUNT = 80


class Cmvn:
    """按照模型训练时的参考范围调整声音特征。"""

    def __init__(self, path: Path) -> None:
        """读取参考数据，并算出后续调整声音特征时需要的数值。"""
        # cmvn.ark 保存了模型训练时统计出的声音数据。这里把文件内容读取成
        # 一个数字表，后面会用它算出每类声音特征的平均值和变化范围。
        stats = kaldiio.load_mat(str(path))

        # 这个数字表应当有两行：第一行记录各项数值的总和，第二行记录平方和。
        # 行数不符合要求，说明文件内容不是当前模型需要的格式。
        if stats.shape[0] != 2:
            raise ValueError(f"CMVN 统计矩阵格式无效：{path}")

        # 最后一列保存参与统计的声音片段数量，前面的列才是声音特征。
        dimension = stats.shape[-1] - 1
        count = stats[0, dimension]

        # 至少要有一个声音片段，才能计算出有效的参考范围。
        if count < 1:
            raise ValueError(f"CMVN 样本数无效：{path}")

        # 这两个列表分别保存每项声音特征的平均值，以及把不同数值范围
        # 调整到相近大小时需要使用的倍数。
        means = []
        inverse_standard_deviations = []

        # 逐项处理声音特征。平均值表示这项数字通常位于哪里，变化程度表示
        # 它平时变化有多大。最小值 1e-20 用来避免变化为零时无法继续计算。
        for index in range(dimension):
            mean = stats[0, index] / count
            variance = max((stats[1, index] / count) - mean * mean, 1e-20)
            means.append(float(mean))
            inverse_standard_deviations.append(1.0 / math.sqrt(variance))

        # 把计算结果保存成模型需要的数字格式，处理每个音频时都可以直接复用。
        self._means = np.asarray(means, dtype=np.float32)
        self._inverse_standard_deviations = np.asarray(
            inverse_standard_deviations,
            dtype=np.float32,
        )

    def apply(self, features: np.ndarray) -> np.ndarray:
        """把新音频的声音特征调整到模型熟悉的数值范围。"""
        # 先减去训练数据的平均值，再按训练数据的变化范围缩放。
        return (features - self._means) * self._inverse_standard_deviations


class FeatureExtractor:
    """负责把音频文件整理成 FireRed 语音识别模型需要的数据。"""

    def __init__(self, cmvn_path: Path) -> None:
        """准备把音频转换成语音识别模型能够理解的数字特征。"""
        # 读取模型训练时使用的音频校准参数，让新音频的数值范围与模型熟悉的
        # 范围保持一致。这样模型才能更稳定地判断声音对应的文字。
        self._cmvn = Cmvn(cmvn_path)

        # 创建一份“如何观察声音”的配置。后面的 extract 函数会使用它分析音频。
        options = knf.FbankOptions()

        # 关闭随机扰动，让同一段音频每次处理都得到相同的结果。
        options.frame_opts.dither = 0.0

        # 分段时只使用完整的声音片段，避免片段边缘被截断后影响识别。
        options.frame_opts.snip_edges = True

        # 每个短声音片段提取 80 个数字，用来描述这段声音的频率特点。
        options.mel_opts.num_bins = MEL_BIN_COUNT

        # 关闭额外的调试输出，正常识别时只保留实际需要的结果。
        options.mel_opts.debug_mel = False

        # 保存配置，处理每个音频文件时重复使用同一套规则。
        self._options = options

    def extract(self, audio_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """读取一个音频文件，返回声音特征和有效片段数量。"""
        # 读取音频中的声音数值和采样率。always_2d=True 会让单声道与多声道
        # 都使用“时间点 × 声道”的统一格式，后面处理起来更简单。
        samples, sample_rate = sf.read(
            str(audio_path),
            dtype="int16",
            always_2d=True,
        )

        # FireRed 模型只接受 16000 Hz 的音频，也就是每秒记录 16000 次声音。
        # 输入不符合要求时立即说明原因，避免模型产生无法理解的结果。
        if sample_rate != EXPECTED_SAMPLE_RATE:
            raise ValueError(
                f"FireRed 输入采样率必须是 {EXPECTED_SAMPLE_RATE} Hz："
                f"path={audio_path} sample_rate={sample_rate}"
            )

        # 把单声道或多声道录音统一整理成一条声道。
        mono_samples = _to_mono(samples)

        # 创建声音分析器，并把完整音频交给它。分析器会按照初始化时保存的
        # 配置，把连续的声音切成许多很短的片段。
        fbank = knf.OnlineFbank(self._options)
        fbank.accept_waveform(sample_rate, mono_samples.astype(np.float32).tolist())

        # 没有得到任何片段，通常表示音频为空或太短，此时无法继续识别。
        if fbank.num_frames_ready == 0:
            raise ValueError(f"FireRed 输入音频没有可用特征帧：{audio_path}")

        # 依次取出每个短声音片段的 80 个特征数字，再按时间顺序拼在一起。
        features = np.vstack(
            [fbank.get_frame(index) for index in range(fbank.num_frames_ready)]
        ).astype(np.float32)

        # 使用前面读取的校准参数调整所有数字，使它们落在模型熟悉的范围内。
        normalized_features = self._cmvn.apply(features).astype(np.float32)

        # 模型一次可以接收多段音频，所以外面再增加一层，表示当前只有一段。
        # 同时返回短声音片段的数量，让模型知道这段数据的有效长度。
        return normalized_features[None, :, :], np.asarray(
            [normalized_features.shape[0]],
            dtype=np.int64,
        )


def _to_mono(samples: np.ndarray) -> np.ndarray:
    """把录音整理成一条声道，方便后续统一分析。"""
    # 音频数据通常是“每个时间点一行、每个声道一列”。只有一条声道时，
    # 直接取出这一列即可，不需要额外处理。
    if samples.shape[1] == 1:
        return samples[:, 0]

    # 立体声等多声道录音会同时保存多列声音。这里对同一时间点的各个声道
    # 求平均值，把左右声道合成一条声音。
    mixed_samples = np.mean(samples.astype(np.int32), axis=1)

    # 把结果限制在音频格式允许的范围内，避免转换成整数时出现异常数值。
    return np.clip(mixed_samples, -32_768, 32_767).astype(np.int16)
