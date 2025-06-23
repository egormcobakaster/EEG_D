# INSTALL AND RUN
python3 -m venv venv

pip install -r requirements.txt

python3 eeg_app.py

# Use
0) FILE->open file->test_data/_3_export.set

1) Установить якорную область на фрагменте с ЭА, например, 174-180 с. Сделать это можно с помощью стрелок навигации и передвижение границ области.

2) Установить treshold на уровне ~0.2.

3) нажать кнопку analyze

4) FILE->open file->test_data/_3_export_matched_174.2-179.9s_thr0.20.set.set

5) выбрать дипозон частот(Fmin, Fmax), построить ЭЭГ-карту (PSD Topomap)