import sys
import os
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import mne
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt, find_peaks
# from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
from torchvision.models import resnet18
from preproc import *
from model import *

# class EEGViewer(QtWidgets.QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("EEG Viewer")
#         self.resize(1200, 800)
#         self.normalizer = NormalizeWindow()
#         self.data = None
#         self.file_path = None
#         self.sampling_rate = None
#         self.current_scale = 1
#         self.window_size = 256
#         self.offset = 0
#         self.window_plot_width = 10  # seconds
#         self.similarity_threshold = 0.9

#         # self.model = EmbeddingModel()
#         self.model=torch.load('../hard_metric/resnet18_19_0.9992762047381143_0.9926679123594849.pt', weights_only=False)
#         self.model.eval()
#         self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#         self.init_ui()

#     def init_ui(self):
#         central_widget = QtWidgets.QWidget()
#         self.setCentralWidget(central_widget)
#         layout = QtWidgets.QVBoxLayout(central_widget)

#         toolbar = QtWidgets.QHBoxLayout()
#         load_btn = QtWidgets.QPushButton("Load File")
#         load_btn.clicked.connect(self.load_file)
#         toolbar.addWidget(load_btn)

#         self.scale_box = QtWidgets.QSpinBox()
#         self.scale_box.setRange(1, 100)
#         self.scale_box.setValue(1)
#         self.scale_box.valueChanged.connect(self.set_scale)
#         toolbar.addWidget(QtWidgets.QLabel("Scale:"))
#         toolbar.addWidget(self.scale_box)

#         self.width_box = QtWidgets.QSpinBox()
#         self.width_box.setRange(1, 60)
#         self.width_box.setValue(10)
#         self.width_box.valueChanged.connect(self.set_width)
#         toolbar.addWidget(QtWidgets.QLabel("Seconds on screen:"))
#         toolbar.addWidget(self.width_box)

#         self.threshold_box = QtWidgets.QDoubleSpinBox()
#         self.threshold_box.setRange(0.0, 1.0)
#         self.threshold_box.setSingleStep(0.01)
#         self.threshold_box.setValue(0.9)
#         self.threshold_box.valueChanged.connect(self.set_threshold)
#         toolbar.addWidget(QtWidgets.QLabel("Threshold:"))
#         toolbar.addWidget(self.threshold_box)

#         analyze_btn = QtWidgets.QPushButton("Analyze")
#         analyze_btn.clicked.connect(self.analyze)
#         toolbar.addWidget(analyze_btn)

#         left_btn = QtWidgets.QPushButton("←")
#         left_btn.clicked.connect(self.scroll_left)
#         toolbar.addWidget(left_btn)

#         right_btn = QtWidgets.QPushButton("→")
#         right_btn.clicked.connect(self.scroll_right)
#         toolbar.addWidget(right_btn)

#         layout.addLayout(toolbar)

#         self.plot_widget = pg.PlotWidget()
#         self.plot_widget.setDownsampling(mode='peak')
#         self.plot_widget.setClipToView(True)
#         self.plot_widget.showGrid(x=True, y=True)
#         layout.addWidget(self.plot_widget)

#         self.region = pg.LinearRegionItem(movable=True)
#         self.plot_widget.addItem(self.region)
#         self.region.setZValue(10)
#         self.region.setRegion([self.offset + 1, self.offset + 2])
#         self.region.sigRegionChanged.connect(self.update_region)

#     def load_file(self):
#         file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open EEG file", "", "EEG Files (*.edf *.set)")
#         if file_path:
#             self.file_path = file_path
#             if file_path.endswith('.edf'):
#                 raw = mne.io.read_raw_edf(file_path, preload=True)
#             elif file_path.endswith('.set'):
#                 raw = mne.io.read_raw_eeglab(file_path, preload=True)
#             else:
#                 return

#             raw.pick_types(eeg=True)
#             self.raw = raw
#             self.data, self.sampling_rate = raw.get_data(), int(raw.info['sfreq'])
#             self.offset = 0
#             self.plot_data()

#     def plot_data(self):
#         self.plot_widget.clear()
#         if self.data is not None:
#             start = int(self.offset * self.sampling_rate)
#             end = int((self.offset + self.window_plot_width) * self.sampling_rate)
#             end = min(end, self.data.shape[1])

#             t = np.arange(start, end) / self.sampling_rate
#             for i in range(self.data.shape[0]):
#                 self.plot_widget.plot(t, self.data[i, start:end] * self.current_scale * 1e6 + i * 100, pen=pg.mkPen(i))

#             self.plot_widget.addItem(self.region)
#             self.region.setRegion([self.offset + 1, self.offset + 2])

#     def set_scale(self, value):
#         self.current_scale = value
#         self.plot_data()

#     def set_width(self, value):
#         self.window_plot_width = value
#         self.plot_data()

#     def set_threshold(self, value):
#         self.similarity_threshold = value

#     def scroll_left(self):
#         self.offset = max(0, self.offset - self.window_plot_width)
#         self.plot_data()

#     def scroll_right(self):
#         if self.data is not None:
#             max_offset = self.data.shape[1] / self.sampling_rate - self.window_plot_width
#             self.offset = min(max_offset, self.offset + self.window_plot_width - 1)
#             self.plot_data()

#     def update_region(self):
#         pass

#     def analyze(self):
#         if self.data is None:
#             return

#         region = self.region.getRegion()
#         start = int(region[0] * self.sampling_rate)
#         end = int(region[1] * self.sampling_rate)
#         if end - start < self.window_size:
#             return

#         # Ключевые окна
#         selected_data = self.data[:, start:end]
#         key_windows = [selected_data[:, i:i + self.window_size] for i in np.linspace(0, selected_data.shape[1] - self.window_size, 3, dtype=int)]
#         print(key_windows[0].shape)
#         key_embeddings = [self.get_embedding(torch.tensor(self.normalizer(w), dtype=torch.float32).unsqueeze(0).unsqueeze(0)) for w in key_windows]

#         # Все окна через DataLoader
#         dataset = EEGWindowDataset(self.data, self.window_size)
#         loader = DataLoader(dataset, batch_size=32)

#         all_embeddings = []
#         window_positions = []

#         with torch.no_grad():
#             for batch in loader:
#                 batch = batch.to(self.device)
#                 embs = self.model(batch).cpu().numpy()
#                 all_embeddings.extend(embs)

#         all_embeddings = np.array(all_embeddings)
#         # key_embeddings = np.array(key_embeddings)

#         # sim = cosine_similarity(key_embeddings, all_embeddings)
#         # l2_distances = []
#         anomaly_segments = []
#         for anomaly_embedding in key_embeddings:
#             l2_distances = np.linalg.norm(all_embeddings - anomaly_embedding, axis=(1))
#             an_seg = detect_anomalies(l2_distances, self.similarity_threshold, margin=0.05, window_size=256, stride=4)
#             anomaly_segments.append(merge_overlapping_segments(an_seg))
#             # anomaly_segments.extend(an_seg)

#         result = intersect_all(anomaly_segments)
#         # result = merge_overlapping_segments(anomaly_segments)
#         results_peaks = process_multichannel_eeg(self.data, self.sampling_rate)
#         slow_wave_peaks_by_channel = {ch: results_peaks[ch]['slow_peaks'] for ch in results_peaks}

#         # Объединяем пики
#         merged_slow_wave_peaks = merge_slow_wave_peaks_across_channels(slow_wave_peaks_by_channel,
#                                                                         min_channels=4,
#                                                                         tolerance=int(0.05 * self.sampling_rate))  # 50 мс


#         # Фильтруем эпилептические сегменты
#         filtered_segments = filter_and_split_epileptic_segments_by_slow_wave_peaks(result, merged_slow_wave_peaks, self.sampling_rate)

#         raw_segments = []
#         for start, end in filtered_segments:
#             raw_segment = self.raw.copy().crop(tmin=start/100, tmax=end/100)
#             raw_segments.append(raw_segment)
        
#         combined_raw = raw_segments[0].copy()
#         for segment in raw_segments[1:]:
#             combined_raw.append(segment)

#         if self.file_path:
#             base, ext = os.path.splitext(self.file_path)
#             region_sec = f"{region[0]:.1f}-{region[1]:.1f}s"
#             save_path = f"{base}_matched_{region_sec}_thr{self.similarity_threshold:.2f}.set"

#             info = mne.create_info(
#                 ch_names=self.raw.info['ch_names'],
#                 sfreq=self.sampling_rate,
#                 ch_types='eeg'
#             )
#             # matched_raw = mne.io.RawArray(combined_raw, info)
#             mne.export.export_raw(save_path, combined_raw, overwrite=True)
#             # mne.export.export_raw(save_path, matched_raw, fmt="auto", overwrite=True)
#             print(f"Saved matched EEG windows to {save_path}")

#     def get_embedding(self, window_tensor):
#         with torch.no_grad():
#             return self.model(window_tensor.to(self.device)).squeeze().cpu().numpy()


# if __name__ == '__main__':
#     app = QtWidgets.QApplication(sys.argv)
#     viewer = EEGViewer()
#     viewer.show()
#     sys.exit(app.exec_())


class EEGViewerTab(QtWidgets.QWidget):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.normalizer = NormalizeWindow()
        self.data = None
        self.sampling_rate = None
        self.current_scale = 1
        self.window_size = 256
        self.offset = 0
        self.window_plot_width = 10  # seconds
        self.similarity_threshold = 0.9
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = torch.load('../hard_metric/resnet18_19_0.9992762047381143_0.9926679123594849.pt', weights_only=False)
        self.model.eval()

        self.init_ui()
        self.load_file()

    def init_ui(self):
    
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

       
        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setSpacing(10)

        
        nav_group = QtWidgets.QGroupBox("Navigation")
        nav_layout = QtWidgets.QHBoxLayout(nav_group)
        
        self.left_btn = QtWidgets.QPushButton("←")
        self.left_btn.setToolTip("Scroll left")
        self.left_btn.clicked.connect(self.scroll_left)
        
        self.right_btn = QtWidgets.QPushButton("→")
        self.right_btn.setToolTip("Scroll right")
        self.right_btn.clicked.connect(self.scroll_right)
        
        nav_layout.addWidget(self.left_btn)
        nav_layout.addWidget(self.right_btn)
        toolbar.addWidget(nav_group)

        
        display_group = QtWidgets.QGroupBox("Display")
        display_layout = QtWidgets.QHBoxLayout(display_group)
        
        self.scale_label = QtWidgets.QLabel("Scale:")
        self.scale_box = QtWidgets.QSpinBox()
        self.scale_box.setRange(1, 100)
        self.scale_box.setValue(1)
        self.scale_box.valueChanged.connect(self.set_scale)
        
        self.width_label = QtWidgets.QLabel("Window (s):")
        self.width_box = QtWidgets.QSpinBox()
        self.width_box.setRange(1, 60)
        self.width_box.setValue(10)
        self.width_box.valueChanged.connect(self.set_width)
        
        display_layout.addWidget(self.scale_label)
        display_layout.addWidget(self.scale_box)
        display_layout.addWidget(self.width_label)
        display_layout.addWidget(self.width_box)
        toolbar.addWidget(display_group)

        
        freq_group = QtWidgets.QGroupBox("Frequency Analysis")
        freq_layout = QtWidgets.QHBoxLayout(freq_group)
        
        self.fmin_label = QtWidgets.QLabel("Fmin (Hz):")
        self.fmin_spin = QtWidgets.QDoubleSpinBox()
        self.fmin_spin.setRange(0.1, 30)
        self.fmin_spin.setValue(8)
        self.fmin_spin.setSingleStep(0.5)
        
        self.fmax_label = QtWidgets.QLabel("Fmax (Hz):")
        self.fmax_spin = QtWidgets.QDoubleSpinBox()
        self.fmax_spin.setRange(0.1, 100)
        self.fmax_spin.setValue(12)
        self.fmax_spin.setSingleStep(0.5)
        
        self.psd_btn = QtWidgets.QPushButton("PSD Topomap")
        self.psd_btn.clicked.connect(self.generate_psd_topomap)
        self.psd_btn.setToolTip("Generate power spectrum topography")
        
        freq_layout.addWidget(self.fmin_label)
        freq_layout.addWidget(self.fmin_spin)
        freq_layout.addWidget(self.fmax_label)
        freq_layout.addWidget(self.fmax_spin)
        freq_layout.addWidget(self.psd_btn)
        toolbar.addWidget(freq_group)

        
        analysis_group = QtWidgets.QGroupBox("Analysis")
        analysis_layout = QtWidgets.QHBoxLayout(analysis_group)
        
        self.threshold_label = QtWidgets.QLabel("Threshold:")
        self.threshold_box = QtWidgets.QDoubleSpinBox()
        self.threshold_box.setRange(0.0, 1.0)
        self.threshold_box.setSingleStep(0.01)
        self.threshold_box.setValue(0.9)
        self.threshold_box.valueChanged.connect(self.set_threshold)
        
        self.analyze_btn = QtWidgets.QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self.analyze)
        
        self.map_btn = QtWidgets.QPushButton("Time Maps")
        self.map_btn.clicked.connect(self.generate_full_file_maps)
        self.map_btn.setToolTip("Generate time-domain topographic maps")
        
        analysis_layout.addWidget(self.threshold_label)
        analysis_layout.addWidget(self.threshold_box)
        analysis_layout.addWidget(self.analyze_btn)
        analysis_layout.addWidget(self.map_btn)
        toolbar.addWidget(analysis_group)

        main_layout.addLayout(toolbar)

        
        self.load_pos_btn = QtWidgets.QPushButton("Load Electrode Positions")
        self.load_pos_btn.clicked.connect(self.load_electrode_positions)
        self.load_pos_btn.setToolTip("Load .set file with electrode positions")
        freq_layout.addWidget(self.load_pos_btn)
        
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        
       
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.scene().sigMouseClicked.connect(self.on_mouse_click)
        self.plot_widget.setBackground('w')
        
        
        self.region = pg.LinearRegionItem([1, 2], movable=True)
        self.region.setZValue(10)
        self.region.setBrush(QtGui.QBrush(QtGui.QColor(0, 255, 0, 50)))
        for line in self.region.lines:
            line.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0, 100)))
        self.plot_widget.addItem(self.region)
        
        
        self.map_widget = pg.GraphicsLayoutWidget()
        self.map_scroll = QtWidgets.QScrollArea()
        self.map_scroll.setWidgetResizable(True)
        self.map_scroll.setWidget(self.map_widget)
        
        self.map_container = QtWidgets.QWidget()
        self.map_layout = QtWidgets.QVBoxLayout(self.map_container)
        self.map_layout.addWidget(self.map_scroll)
        self.map_container.hide()
        
        splitter.addWidget(self.plot_widget)
        splitter.addWidget(self.map_container)
        splitter.setSizes([400, 200])
        main_layout.addWidget(splitter)

        
        self.status_bar = QtWidgets.QStatusBar()
        self.status_bar.setSizeGripEnabled(False)
        main_layout.addWidget(self.status_bar)

        
        self.setStyleSheet("""
            QGroupBox {
                border: 1px solid gray;
                border-radius: 3px;
                margin-top: 0.5em;
                padding: 5px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
            QPushButton {
                padding: 3px 8px;
                min-width: 60px;
            }
            QSpinBox, QDoubleSpinBox {
                min-width: 60px;
                max-width: 80px;
            }
        """)

    def generate_full_file_maps(self):
        
        if self.data is None or not hasattr(self, 'montage') or self.montage is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No EEG data or montage loaded")
            return

        
        map_interval = 1.0  
        samples_per_map = int(map_interval * self.sampling_rate)
        total_maps = self.data.shape[1] // samples_per_map

        
        self.map_widget.clear()
        self.map_container.show()

        
        rows = 4
        cols = min(4, (total_maps + rows - 1) // rows)
        
        
        for i in range(min(16, total_maps)): 
            start = i * samples_per_map
            end = start + samples_per_map
            avg_data = np.mean(self.data[:, start:end], axis=1)
            
            
            if i > 0 and i % cols == 0:
                self.map_widget.nextRow()
            
            plt = self.map_widget.addPlot(title=f"Time: {i*map_interval:.1f}s")
            
            
            im = self.create_topomap(avg_data, plt)
            
            
            if i == 0:
                bar = pg.ColorBarItem(values=(im.min(), im.max()), colorMap='viridis')
                bar.setImageItem(im)
                bar.setOrientation('h')
                self.map_widget.addItem(bar)

    def load_electrode_positions(self):
        
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Electrode Positions File", "", "EEG Files (*.set)"
        )
        if not file_path:
            return

        try:
            
            raw_with_pos = mne.io.read_raw_eeglab(file_path, preload=False)
            
            if not hasattr(self, 'raw'):
                QtWidgets.QMessageBox.warning(self, "Warning", "Load EEG data first")
                return

            
            print("\n=== DEBUG CHANNEL NAMES ===")
            print("Data channels:", self.raw.ch_names)
            print("Position file channels:", raw_with_pos.ch_names)
            
            
            montage = raw_with_pos.get_montage()
            if montage is None:
                QtWidgets.QMessageBox.warning(self, "Warning", "No positions in file")
                return

            
            pos_dict = montage.get_positions()['ch_pos']
            print("Montage channels:", list(pos_dict.keys()))
            
            
            def transform_ch_name(ch_name):
                ch_name = ch_name.replace('-Pseudo', '')
                ch_name = ch_name.replace('EEG ', '')
                
                ch_name = ch_name.replace('T3', 'T7').replace('T4', 'T8').replace('T5', 'P7').replace('T6', 'P8')
                return ch_name
            
            ch_mapping = {}
            for ch in self.raw.ch_names:
                transformed_name = transform_ch_name(ch)
                print(transformed_name, pos_dict.keys())
                if transformed_name in pos_dict:
                    ch_mapping[ch] = pos_dict[transformed_name]
            
            print("Matched channels:", ch_mapping.keys())
            
            if not ch_mapping:
                QtWidgets.QMessageBox.warning(
                    self, "Warning", 
                    "No matching channels found after transformations.\n"
                    "Data channels:\n" + "\n".join(self.raw.ch_names) + "\n\n"
                    "Position channels:\n" + "\n".join(pos_dict.keys())
                )
                return

            
            from mne.channels import make_dig_montage
            self.modified_montage = make_dig_montage(
                ch_pos=ch_mapping,
                coord_frame='head'
            )

            
            self.raw.set_montage(self.modified_montage, on_missing='warn')
            
            
            print("Applied montage to:", [ch for ch in self.raw.ch_names if ch in ch_mapping])
            print("Missing positions for:", [ch for ch in self.raw.ch_names if ch not in ch_mapping])
            
            
            applied = len([ch for ch in self.raw.ch_names if ch in ch_mapping])
            total = len(self.raw.ch_names)
            missing = [ch for ch in self.raw.ch_names if ch not in ch_mapping]
            
            msg = f"Applied positions for {applied}/{total} channels\n"
            if missing:
                msg += f"\nMissing positions for:\n{', '.join(missing)}"
            
            QtWidgets.QMessageBox.information(
                self, "Channel Mapping Report", msg
            )

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", 
                f"Failed to load positions:\n{str(e)}\n"
                f"Data channels: {self.raw.ch_names if hasattr(self, 'raw') else 'N/A'}\n"
                f"Position channels: {raw_with_pos.ch_names if 'raw_with_pos' in locals() else 'N/A'}"
            )

    def generate_psd_topomap(self):
        
        if not hasattr(self, 'raw') or self.raw is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No EEG data loaded")
            return
        
        try:
           
            print("\n=== TOPOMAP GENERATION DEBUG ===")
            print("Original channels:", self.raw.ch_names)
            
            
            raw_temp = self.raw.copy()
            
            
            if hasattr(self, 'modified_montage'):
                print("Using modified montage channels:", self.modified_montage.ch_names)
                available_ch = [ch for ch in self.modified_montage.ch_names if ch in raw_temp.ch_names]
                print("Available channels:", available_ch)
                raw_temp.pick_channels(available_ch)
            else:
                print("No montage found, trying automatic name matching")
                raw_temp.rename_channels(lambda x: x.replace('-Pseudo', ''))
            
            print("Channels for topomap:", raw_temp.ch_names)
            
            
            if raw_temp.info['dig'] is None:
                QtWidgets.QMessageBox.warning(
                    self, "Warning", 
                    "No electrode positions found for:\n" + "\n".join(raw_temp.ch_names)
                )
                return
            
            
            fmin = self.fmin_spin.value()
            fmax = self.fmax_spin.value()
            print(f"Calculating PSD for {fmin}-{fmax} Hz")
            
            spectrum = raw_temp.compute_psd(method='welch', fmin=fmin, fmax=fmax, n_fft=2048)
            psd_avg = spectrum.get_data().mean(axis=-1)
            print("PSD calculation completed")
            
            
            topo_window = QtWidgets.QMainWindow(self)
            topo_window.setWindowTitle(f"PSD Topomap {fmin}-{fmax} Hz - {len(raw_temp.ch_names)} channels")
            
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            
            fig = Figure(figsize=(10, 8))
            canvas = FigureCanvasQTAgg(fig)
            topo_window.setCentralWidget(canvas)
            
            ax = fig.add_subplot(111)
            im, _ = mne.viz.plot_topomap(
                psd_avg,
                raw_temp.info,
                ch_type='eeg',
                cmap='Spectral_r',
                axes=ax,
                show=False,
                sensors=True,
                outlines='head'
            )
            
            
            ax.set_title(
                f"PSD {fmin}-{fmax} Hz\n"
                f"Channels: {', '.join(raw_temp.ch_names)}",
                fontsize=10
            )
            fig.colorbar(im, ax=ax).set_label('Power (dB)')
            
            canvas.draw()
            topo_window.resize(1000, 800)
            topo_window.show()
            
            print("Topomap generated successfully")

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", 
                f"Topomap generation failed:\n{str(e)}\n"
                f"Channels attempted: {raw_temp.ch_names if 'raw_temp' in locals() else 'N/A'}\n"
                f"Check console for detailed debug output"
            )
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()

    
    def create_topomap(self, data, plot_item):
       
        from scipy.interpolate import griddata
        
        
        x, y = self.pos[:, :2].T
        xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
        
        
        zi = griddata((x, y), data, (xi, yi), method='cubic')
        
        
        img = pg.ImageItem()
        img.setImage(zi.T)
        plot_item.addItem(img)
        plot_item.setAspectLocked(True)
        plot_item.hideAxis('bottom')
        plot_item.hideAxis('left')
        
        return img

    def load_file(self):
        if self.file_path.endswith('.edf'):
            raw = mne.io.read_raw_edf(self.file_path, preload=True)
        elif self.file_path.endswith('.set'):
            raw = mne.io.read_raw_eeglab(self.file_path, preload=True)
        else:
            return

        raw.pick_types(eeg=True)
        self.raw = raw
        self.data, self.sampling_rate = raw.get_data(), int(raw.info['sfreq'])
        self.ch_names = raw.info['ch_names']
        
        # Получаем позиции электродов
        if raw.info['dig'] is not None:
            self.montage = raw.get_montage()
            if self.montage:
                self.pos = self.montage.get_positions()['ch_pos']
                self.pos = np.array([self.pos[ch] for ch in self.ch_names if ch in self.pos])
        
        self.offset = 0
        self.plot_data()

    def plot_data(self):
        self.plot_widget.clear()
        if self.data is not None:
            start = int(self.offset * self.sampling_rate)
            end = int((self.offset + self.window_plot_width) * self.sampling_rate)
            end = min(end, self.data.shape[1])
            t = np.arange(start, end) / self.sampling_rate
            for i in range(self.data.shape[0]):
                self.plot_widget.plot(t, self.data[i, start:end] * self.current_scale * 1e6 + i * 100, pen=pg.mkPen(i))
            self.plot_widget.addItem(self.region)

    def on_mouse_click(self, event):
        if event.double():
            pos = self.plot_widget.plotItem.vb.mapSceneToView(event.scenePos()).x()
            self.region.setRegion([pos, pos + 1])

    def set_scale(self, value):
        self.current_scale = value
        self.plot_data()

    def set_width(self, value):
        self.window_plot_width = value
        self.plot_data()

    def set_threshold(self, value):
        self.similarity_threshold = value

    def scroll_left(self):
        prev_offset = self.offset
        self.offset = max(0, self.offset - self.window_plot_width)
        delta = prev_offset - self.offset
        self.shift_region(-delta)
        self.plot_data()

    def scroll_right(self):
        prev_offset = self.offset
        max_offset = max(0, (self.data.shape[1] / self.sampling_rate) - self.window_plot_width)
        self.offset = min(max_offset, self.offset + self.window_plot_width - 1)
        delta = self.offset - prev_offset
        self.shift_region(delta)
        self.plot_data()

    def shift_region(self, delta_seconds):
        
        region = self.region.getRegion()
        self.region.setRegion([r + delta_seconds for r in region])


    def analyze(self):
        print("analyze")
        if self.data is None:
            return

        region = self.region.getRegion()
        start = int(region[0] * self.sampling_rate)
        end = int(region[1] * self.sampling_rate)
        if end - start < self.window_size:
            return

        selected_data = self.data[:, start:end]
        key_windows = [selected_data[:, i:i + self.window_size] for i in np.linspace(0, selected_data.shape[1] - self.window_size, 3, dtype=int)]
        key_embeddings = [self.get_embedding(torch.tensor(self.normalizer(w), dtype=torch.float32).unsqueeze(0).unsqueeze(0)) for w in key_windows]

        dataset = EEGWindowDataset(self.data, self.window_size)
        loader = DataLoader(dataset, batch_size=32)

        all_embeddings = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                embs = self.model(batch).cpu().numpy()
                all_embeddings.extend(embs)

        all_embeddings = np.array(all_embeddings)
        anomaly_segments = []
        for anomaly_embedding in key_embeddings:
            l2_distances = np.linalg.norm(all_embeddings - anomaly_embedding, axis=1)
            an_seg = detect_anomalies(l2_distances, self.similarity_threshold, margin=0.05, window_size=256, stride=4)
            anomaly_segments.append(merge_overlapping_segments(an_seg))

        result = intersect_all(anomaly_segments)
        results_peaks = process_multichannel_eeg(self.data, self.sampling_rate)
        slow_wave_peaks_by_channel = {ch: results_peaks[ch]['slow_peaks'] for ch in results_peaks}
        merged_slow_wave_peaks = merge_slow_wave_peaks_across_channels(slow_wave_peaks_by_channel, min_channels=4, tolerance=int(0.05 * self.sampling_rate))
        filtered_segments = filter_and_split_epileptic_segments_by_slow_wave_peaks(result, merged_slow_wave_peaks, self.sampling_rate)

        raw_segments = []
        for start, end in filtered_segments:
            raw_segment = self.raw.copy().crop(tmin=start/100, tmax=end/100)
            raw_segments.append(raw_segment)

        combined_raw = raw_segments[0].copy()
        for segment in raw_segments[1:]:
            combined_raw.append(segment)

        base, ext = os.path.splitext(self.file_path)
        region_sec = f"{region[0]:.1f}-{region[1]:.1f}s"
        save_path = f"{base}_matched_{region_sec}_thr{self.similarity_threshold:.2f}.set"
        mne.export.export_raw(save_path, combined_raw, overwrite=True)
        print(f"Saved matched EEG windows to {save_path}")

    def get_embedding(self, window_tensor):
        with torch.no_grad():
            return self.model(window_tensor.to(self.device)).squeeze().cpu().numpy()


class EEGViewerMain(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Viewer with Tabs")
        self.resize(1200, 800)

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        self.init_menu()

    def init_menu(self):
        open_action = QtWidgets.QAction("Open File", self)
        open_action.triggered.connect(self.open_file)
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        file_menu.addAction(open_action)

    def open_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open EEG file", "", "EEG Files (*.edf *.set)")
        if file_path:
            tab = EEGViewerTab(file_path)
            self.tabs.addTab(tab, os.path.basename(file_path))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    viewer = EEGViewerMain()
    viewer.show()
    sys.exit(app.exec_())

