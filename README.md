# DDPM: Diffusion Models + C++ Inference Engine

Пет-проект: семейство диффузионных моделей (от 2D-гауссов до генерации фото)
и собственный C++ движок для инференса.

## Структура

```
models/   — Python/PyTorch: обучение и эксперименты
engine/   — C++ движок инференса (CPU SIMD → CUDA)
format/   — формат весов .ddpm + конвертеры из PyTorch
notebooks/— теория и визуализации
```

## Лестница моделей

1. Toy: 2D-смеси гауссов (быстрое понимание процесса)
2. MNIST 28×28 (базовый DDPM/DDIM, первый parity-тест движка)
3. 64×64 с класс-кондиционированием (CFG)
4. Latent Diffusion (VAE + LDM)
5. Stretch: flow matching, SD-mini

## Статус

- [x] Skeleton репозитория
- [ ] Toy DDPM
- [ ] MNIST DDPM
- [ ] Движок v0: формат весов + CPU ops
- [ ] Parity-тест CPU vs PyTorch
- [ ] CUDA backend v0

## Правила

- Обучение на GPU запускает только владелец (не агенты/CI)
- Каждый этап модели = self-contained: скрипты + конфиг + README
- Движок: scalar-реализация каждой op = эталон для тестов
