# Proximal GaLore: Динамическая адаптация ранга градиента через регуляризацию ядерной нормой

Участники проекта:

Виноградов Николай Александрович (Б05-421)
Агафонкин Егор Дмитриевич (Б05-421)
Гжибовский Арсений Сергеевич (Б05-421)

---

## Описание проекта

Реализация и экспериментальное сравнение трёх оптимизаторов на основе AdamW:

| Метод | Описание |
|---|---|
| **AdamW** | Стандартный AdamW (baseline) |
| **GaLore AdamW** | AdamW с проекцией градиента на подпространство фиксированного ранга через Truncated SVD ([Zhao et al., 2024](https://arxiv.org/abs/2403.03507)) |
| **Proximal GaLore AdamW** | AdamW с динамической адаптацией ранга через Singular Value Thresholding (проксимальный оператор ядерной нормы) |

### Ключевая идея

В оригинальном GaLore ранг проекции `r` задаётся вручную. Мы предлагаем заменить жёсткое усечение (Truncated SVD) на **мягкое пороговое отсечение** сингулярных чисел (SVT):

```
σ_i → max(σ_i − λ, 0)
```

Это проксимальный оператор ядерной нормы, который автоматически определяет эффективный ранг подпространства на каждом шаге пересчёта SVD.

## Структура проекта

```
galore/
├── README.md                    # Этот файл
├── requirements.txt             # Зависимости
├── experiment.py                # Основной эксперимент + визуализации
├── galore_framework/            # Фреймворк
│   ├── __init__.py
│   ├── projector.py             # GaLoreProjector, ProximalGaLoreProjector
│   ├── optimizers.py            # StandardAdamW, GaLoreAdamW, ProximalGaLoreAdamW
│   └── utils.py                 # TrainingTracker, метрики, утилиты
├── results/                     # Графики (генерируются автоматически)
└── pdf-files/
    └── idea.pdf                 # Описание идеи проекта
```

## Установка и запуск

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Запуск эксперимента

```bash
python3 experiment.py
```

Скрипт:
- Генерирует синтетический датасет (4096 примеров, 256 признаков, 10 классов)
- Обучает MLP (256→512→512→256→10) тремя оптимизаторами
- Сохраняет графики в директорию `results/`

### 3. Результаты

После запуска в папке `results/` появятся следующие графики:

| Файл | Содержание |
|---|---|
| `loss_curves.png` | Кривые обучения всех трёх методов |
| `rank_evolution.png` | Эволюция эффективного ранга по эпохам |
| `memory_footprint.png` | Потребление памяти состояниями оптимизатора |
| `singular_values.png` | Распределение сингулярных чисел градиента (начало vs конец обучения) |
| `grad_norms.png` | Норма градиента по шагам обучения |
| `proximal_rank_dynamics.png` | Детальная динамика ранга для Proximal GaLore |

## Ключевые параметры

| Параметр | Значение | Описание |
|---|---|---|
| `GALORE_RANK` | 32 | Фиксированный ранг для GaLore |
| `SVT_THRESHOLD` | 0.03 | Порог мягкого отсечения (λ) для Proximal GaLore |
| `UPDATE_PROJ_GAP` | 50 | Частота пересчёта SVD (в шагах) |
| `MIN_RANK` | 4 | Минимальный ранг для Proximal GaLore |

## Фреймворк `galore_framework`

### Проекторы (`projector.py`)

```python
from galore_framework import GaLoreProjector, ProximalGaLoreProjector

# Фиксированный ранг (оригинальный GaLore)
proj = GaLoreProjector(rank=32, update_freq=200)

# Динамический ранг (Proximal GaLore с SVT)
proj = ProximalGaLoreProjector(threshold=0.03, update_freq=200, min_rank=1)

# Использование
low_rank_grad = proj.project(full_gradient)
full_update = proj.project_back(low_rank_update)
print(proj.get_effective_rank())
```

### Оптимизаторы (`optimizers.py`)

```python
from galore_framework import StandardAdamW, GaLoreAdamW, ProximalGaLoreAdamW

# Baseline
opt = StandardAdamW(model.parameters(), lr=1e-3)

# GaLore (фиксированный ранг)
opt = GaLoreAdamW(model.parameters(), lr=1e-3, rank=32, update_proj_gap=200)

# Proximal GaLore (динамический ранг через ядерную норму)
opt = ProximalGaLoreAdamW(model.parameters(), lr=1e-3, threshold=0.03, min_rank=4)
```