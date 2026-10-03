"""Linear beta schedule — базовая схема DDPM (Ho et al., 2020).

В DDPM есть T шагов. На каждом шаге t мы знаем:
  - beta_t  — сколько НОВОГО шума добавить на этом шаге
  - alpha_t = 1 - beta_t
  - alpha_bar_t = prod_{s=1..t} alpha_s — накопленная "чистота"

И главное свойство: можно добавить шум на T шагов сразу,
без итераций: x_t = sqrt(a_bar_t) * x_0 + sqrt(1 - a_bar_t) * eps.
Это и делает обучение быстрым: берём случайный t и один шаг.
"""
from __future__ import annotations

import torch


class LinearBetaSchedule:
    def __init__(self, num_steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.num_steps = num_steps
        # beta: 1e-4 → 0.02 линейно. На ранних шагах почти нет шума,
        # на поздних — много. alpha_bar_1000 ≈ 0.084 → почти чистый шум.
        self.beta = torch.linspace(beta_start, beta_end, num_steps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def to(self, device) -> "LinearBetaSchedule":
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        return self

    def x_t_from_x_0(self, x_0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor | None = None) -> torch.Tensor:
        """Рекуррентный forward process в один прыжок (см. docstring модуля)."""
        if eps is None:
            eps = torch.randn_like(x_0)
        abar = self.alpha_bar[t].view(-1, 1)  # (B,) → (B,1) для бродкаста
        return (abar.sqrt() * x_0 + (1 - abar).sqrt() * eps, eps)

    def training_targets(self, x_0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Сэмплируем случайные t и возвращаем (x_t, target).
        Целевое значение в оригинальном DDPM — именно eps (предсказание шума),
        поэтому модель называется epsilon-predictor.
        """
        b = x_0.shape[0]
        t = torch.randint(0, self.num_steps, (b,), device=x_0.device)
        x_t, eps = self.x_t_from_x_0(x_0, t)
        return x_t, t, eps
