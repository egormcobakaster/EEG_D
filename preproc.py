
import mne
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt, find_peaks
import torch.nn as nn
from torchvision.models import resnet18

def intersect_intervals(intervals1, intervals2):
    """Находит пересечение двух списков интервалов"""
    result = []
    i = j = 0

    while i < len(intervals1) and j < len(intervals2):
        a_start, a_end = intervals1[i]
        b_start, b_end = intervals2[j]

        # Находим пересечение
        start = max(a_start, b_start)
        end = min(a_end, b_end)

        if start < end:  # Есть пересечение
            result.append([start, end])

        # Продвигаемся по тому интервалу, который заканчивается раньше
        if a_end < b_end:
            i += 1
        else:
            j += 1

    return result

def detect_slow_waves(data, fs, threshold, min_duration_ms=100, max_duration_ms=500):
    """
    Детекция «медленных» волн (slow waves).
    
    threshold: амплитудный порог
    min_duration_ms, max_duration_ms: допустимый диапазон длительности (мс)
    
    Возвращает:
    - valid_peaks: массив индексов обнаруженных пиков, соответствующих критериям.
    """
    # Поиск пиков, превышающих порог
    # peaks_w, peak_props_w = find_peaks(data, prominence=threshold)
    
    # Используем ширину пика на 50% амплитуды для оценки длительности
    peaks_w, peak_props_w = find_peaks(data, prominence=threshold, width=1, rel_height=0.5)
    
    min_samples = (min_duration_ms / 1000.) * fs
    max_samples = (max_duration_ms / 1000.) * fs
    
    valid_peaks = []
    for i, peak in enumerate(peaks_w):
        width_in_samples = peak_props_w['widths'][i]
        if (width_in_samples >= min_samples) and (width_in_samples <= max_samples):
            valid_peaks.append(peak)
    
    return np.array(valid_peaks, dtype=int)

def process_multichannel_eeg(eeg_data, fs):
    """
    Обработка многоканальных ЭЭГ-данных.
    
    eeg_data: матрица с размерностью [n_channels, n_samples]
    fs: частота дискретизации (Гц)
    
    На каждом канале применяем фильтрацию и детекцию sharp и slow волн.
    Возвращаем словарь с результатами для каждого канала.
    """
    # Задайте параметры фильтрации
    lowcut = 0.5
    highcut = 40
    
    # Параметры детекции. Возможна индивидуальная настройка для каждого типа волн.
    sharp_threshold = 5
    slow_threshold = 50

    results = {}
    n_channels = eeg_data.shape[0]
    
    for ch in range(n_channels):
        # Получаем данные конкретного канала
        data = eeg_data[ch, :]
        
        # Фильтрация сигнала для текущего канала
        filt_data = bandpass_filter(data, lowcut, highcut, fs, order=4) * 1e6
        
        # Детектируем sharp и slow волны
        # sharp_peaks = detect_sharp_waves(filt_data, fs,
        #                                  threshold=sharp_threshold,
        #                                  min_duration_ms=5, 
        #                                  max_duration_ms=100)
        
        slow_peaks = detect_slow_waves(filt_data, fs,
                                       threshold=slow_threshold,
                                       min_duration_ms=100, 
                                       max_duration_ms=400)
        # Сохраняем результаты для текущего канала
        results[ch] = {
            'filtered_data': filt_data,
            # 'sharp_peaks': sharp_peaks,
            'slow_peaks': slow_peaks
        }
    
    return results

def filter_and_split_epileptic_segments_by_slow_wave_peaks(epileptic_segments, slow_wave_peaks, fs,
                                                            window_ms=400):
    """
    уточняет эпилептические сегменты по содержанию slow волны.
    
    epileptic_segments: список [(start, end)]
    slow_wave_peaks: список пиков (объединённых по всем каналам!)
    fs: частота дискретизации
    window_ms: ширина окна вокруг пика, определяющего slow волну
    """
    window_samples = int((window_ms / 1000) * fs)
    half_window = window_samples // 2

    from bisect import bisect_left, bisect_right

    # Сортируем пики один раз
    slow_wave_peaks = sorted(slow_wave_peaks)

    result_segments = []

    for seg_start, seg_end in epileptic_segments:
        # Найдём индексы пиков, попадающих в сегмент
        left = bisect_left(slow_wave_peaks, seg_start - half_window)
        right = bisect_right(slow_wave_peaks, seg_end + half_window)
        peaks_in_segment = slow_wave_peaks[left:right]

        if not peaks_in_segment:
            continue  # Ничего не добавляем

        # Создаем окна вокруг каждого пика
        peak_windows = [(max(seg_start, peak - half_window),
                         min(seg_end, peak + half_window)) for peak in peaks_in_segment]

        # Объединяем перекрывающиеся окна
        merged_intervals = merge_intervals(peak_windows)

        result_segments.extend(merged_intervals)
        print(seg_start, seg_end, merged_intervals)

    return result_segments


def merge_intervals(intervals):
    """Объединяет перекрывающиеся интервалы."""
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged



def union_all(list_of_interval_lists):
    """Объединяет все списки интервалов"""
    if not list_of_interval_lists:
        return []

    result = []
    for interval_list in list_of_interval_lists:
        result.extend(interval_list)

    return merge_overlapping_segments(result)


def intersect_all(list_of_interval_lists):
    """Пересекает все списки интервалов"""
    if not list_of_interval_lists:
        return []

    result = list_of_interval_lists[0]
    for next_list in list_of_interval_lists[1:]:
        result = intersect_intervals(result, next_list)
        if not result:
            break  # Нет пересечений

    return result


def detect_anomalies(probabilities, threshold, margin=0.05, window_size=256, stride=32):
    anomalies = []
    start = None

    for idx, prob in enumerate(probabilities):
        i = idx * stride  # Текущее реальное смещение
        if prob < threshold:
            if start is None:
                start = i
        else:
            if start is not None and prob > threshold + margin:
                if i - start >= window_size - stride:  # Учитываем stride при проверке длины
                    anomalies.append((start, i + window_size))
                    start = None

    if start is not None:
        anomalies.append((start, (len(probabilities) - 1) * stride + window_size))

    return anomalies

import numpy as np
from collections import defaultdict

def merge_slow_wave_peaks_across_channels(slow_wave_peaks_by_channel, min_channels=2, tolerance=50):
    """
    Объединяет пики slow волн по всем каналам в глобальные пики.

    slow_wave_peaks_by_channel: словарь {channel: [peak1, peak2, ...]}
    min_channels: минимальное число каналов, в которых пик должен быть замечен
    tolerance: допустимая разница по времени (в сэмплах) между пиками в разных каналах
    """
    all_peaks = []

    # Собираем все пики с указанием канала
    for ch, peaks in slow_wave_peaks_by_channel.items():
        for p in peaks:
            all_peaks.append((p, ch))

    # Сортируем по времени
    all_peaks.sort(key=lambda x: x[0])

    merged_peaks = []
    used = [False] * len(all_peaks)

    for i in range(len(all_peaks)):
        if used[i]:
            continue

        center, ch_i = all_peaks[i]
        cluster = [(center, ch_i)]
        used[i] = True

        # Сравниваем с последующими пиками
        for j in range(i + 1, len(all_peaks)):
            if used[j]:
                continue
            t, ch_j = all_peaks[j]
            if abs(t - center) <= tolerance:
                cluster.append((t, ch_j))
                used[j] = True
            elif t - center > tolerance:
                break  # можно прервать, так как список отсортирован

        involved_channels = set(ch for _, ch in cluster)
        if len(involved_channels) >= min_channels:
            # Добавляем усредненное значение пиков в кластер
            avg_peak = int(np.mean([t for t, _ in cluster]))
            merged_peaks.append(avg_peak)

    return sorted(merged_peaks)

from scipy.signal import butter, filtfilt, find_peaks

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Простейший Баттерворта фильтр в полосе [lowcut, highcut].
    
    data: 1D массив ЭЭГ-сигнала (один канал)
    lowcut, highcut: частоты среза (Гц)
    fs: частота дискретизации (Гц)
    order: порядок фильтра
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def merge_overlapping_segments(segments):
    if not segments:
        return []
    
    # Сортируем сегменты по началу
    segments.sort(key=lambda x: x[0])
    merged = [segments[0]]
    
    for current in segments[1:]:
        previous = merged[-1]
        # Если текущий сегмент пересекается или соприкасается с предыдущим
        if current[0] <= previous[1]:
            # Объединяем сегменты
            merged[-1] = (previous[0], max(previous[1], current[1]))
        else:
            # Добавляем как новый сегмент
            merged.append(current)
    
    return merged