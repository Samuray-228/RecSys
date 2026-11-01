# Рекомендательные системы 

**Студент:** Леонтьев Константин Валерьевич

## ДЗ 1

Пять моделей top-10 рекомендаций, реализованных с нуля (NumPy / SciPy / PyTorch), на датасете
Goodreads Comics & Graphic (UCSD Book Graph):

| модель | файл | NDCG@10 (test) |
|---|---|---|
| EASE | `models/ease.py` | **0.1107** |
| Item-based KNN | `models/item_knn.py` | 0.1039 |
| iALS с собственным таргетом (оценка × давность) | `models/ials_custom.py` | 0.1025 |
| iALS | `models/ials.py` | 0.0720 |
| Factorization Machine с метапризнаками | `models/fm.py` | 0.0526 |

* [`hw1/experiments.ipynb`](hw1/experiments.ipynb) — основной ноутбук с выполненными ячейками
  (данные, EDA, протокол, подбор на validation, сравнение на test, выводы);
* [`hw1/DATA.md`](hw1/DATA.md) — датасет и ссылки на источник;
* [`hw1/report.pdf`](hw1/report.pdf) — отчёт (LaTeX-исходник рядом);
* [`hw1/requirements.txt`](hw1/requirements.txt) — зависимости.

Запуск: `pip install -r hw1/requirements.txt`, затем выполнить `hw1/experiments.ipynb` сверху вниз из папки `hw1`.
