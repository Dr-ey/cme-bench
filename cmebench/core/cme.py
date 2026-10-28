# Ported from cat_vs_dog/exp_001_cgd.py (exp_001, article version 26.10). Logic unchanged; imports rewritten.
"""exp_001 — extended CME with a selectable class-loss variant.

Wraps the CME optimization (same math as vit_degradation.optimize_alpha) but adds a `loss_mode`:

  * "logits"  — MSE over the FULL logit vector (the original behavior). Preserves the whole
                output; tends to produce diffuse masks (see doc/comparison_protocol.md analysis).
  * "margin"  — preserves only the decision MARGIN of the predicted class
                (logit[target] - logit[other]). Should yield more SELECTIVE masks because it
                only protects the decision, not every logit -> can drop more of the image.

`vit_degradation.py` is left untouched; we reuse its `_smoothness_loss`. This module is the SINGLE
SOURCE OF TRUTH for the CME configuration (CGD_CONFIG / CGD_LOSS_MODE / CGD_MAX_AREA) and owns
`binarize_alpha_2d`. `loss_mode="margin"` is binary-only (2 classes).

Used by exp_001_orchestrator.ipynb, exp_001_experiment.py, and the recompute scripts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from cmebench.core.degradation import DegradationConfig, _smoothness_loss

LossMode = Literal["logits", "margin", "target"]

# --------------------------------------------------------------------------- #
# КАНОНІЧНА КОНФІГУРАЦІЯ CME — ЄДИНЕ ДЖЕРЕЛО ПРАВДИ для всіх файлів проєкту.
# Узгоджено (doc/comparison_protocol.md §5.1): baseline = СЕРЕДНІЙ КОЛІР; loss = logits;
# кеп значимої зони = 20%; max ітерацій = 1000; lr0.05 wc0.05 wb0.95 ws0.5.
# Усі entry-points (recompute-скрипти, orchestrator, run_experiment) спираються на це.
# --------------------------------------------------------------------------- #
CGD_LOSS_MODE: LossMode = "logits"  # "logits" | "margin" (margin only for binary)
CGD_MAX_AREA: float = 0.20
CGD_CONFIG = DegradationConfig(
    iterations=1000,
    lr=0.05,
    weight_class=0.05,
    weight_blur=0.95,
    weight_sharpness=0.5,
    err_method="square",
    init_alpha_logit=4.0,
)


def binarize_alpha_2d(alpha_2d: np.ndarray, max_area: float = CGD_MAX_AREA) -> Tuple[np.ndarray, float]:
    """Blur + adaptive threshold, capped so the significant zone never exceeds `max_area`.

    Base rule (as in the CME notebook): threshold = mean - (max-min)/50 on the blurred alpha.
    Cap: if that yields area > max_area, raise the threshold to keep exactly the top `max_area`
    fraction of pixels (strict top-k). Returns (binary_mask bool [H,W], threshold).
    """
    alpha_blur = cv2.GaussianBlur(alpha_2d.astype(np.float32), (5, 5), 0)
    a_mean, a_min, a_max = float(alpha_blur.mean()), float(alpha_blur.min()), float(alpha_blur.max())
    threshold = float(np.clip(a_mean - (a_max - a_min) / 50.0, a_min, a_max))
    binary = alpha_blur >= threshold
    if binary.mean() > max_area:
        # Strict cap: keep exactly the top floor(max_area * N) pixels (ties cannot push area over).
        flat = alpha_blur.reshape(-1)
        k = int(np.floor(max_area * flat.size))
        top_idx = np.argpartition(flat, flat.size - k)[flat.size - k:]
        binary = np.zeros(flat.size, dtype=bool)
        binary[top_idx] = True
        binary = binary.reshape(alpha_blur.shape)
        threshold = float(flat[top_idx].min())
    return binary, threshold
StopRule = Literal["any", "all", "patience", "rel"]


@dataclass
class EarlyStopConfig:
    """Критерій ранньої зупинки CME за уповільненням повної втрати (доповнює `max_iter`).

    Формалізує "занадто повільно" двома НЕЗАЛЕЖНИМИ умовами плато total_loss L_t. Обидві —
    адаптивні *по зображенню* (не по архітектурі), тож НЕ порушують architecture-agnostic
    (doc/comparison_protocol.md §5.1; пор. формалізацію doc/math.md §1):

      (a) patience — best L не покращився більше ніж на `min_delta` протягом `patience` ітерацій:
              no_improve(t) ≥ patience,  де покращення = (best − L_t > min_delta).
          ⚠️ `min_delta` АБСОЛЮТНИЙ → залежить від масштабу втрати (тому за замовч. варіант (b)/'any').

      (b) rel — середня |ΔL| за останнє вікно `window` впала в `rel_factor` (X) разів проти
          середньої |ΔL| ПЕРШОГО вікна (початкова швидкість спуску):
              r_t = (1/W)·Σ|L_k − L_{k-1}|  (останні W),   r_init = той самий показник на перших W кроках,
              стоп, якщо r_t < r_init / X.   ← саме «зміна стала в X разів меншою від початкової».

    `rule`: 'patience' (лише a) | 'rel' (лише b) | 'any' (a АБО b, за замовч.) | 'all' (a І b).
    `warmup` (≥ `window`) — мін. ітерацій до першої можливої зупинки. `enabled=False` → вимкнено.

    Зв'язок із doc/math.md §1: там плато задано ВІДНОСНО ПОТОЧНОЇ величини втрати
    (|L_{t−p}−L_t|/max(|L_{t−p}|,δ) < ε_L); тут (b) задано ВІДНОСНО ПОЧАТКОВОЇ ШВИДКОСТІ. Це
    свідомо інша (запропонована) формалізація — узгодити math.md з кодом при фіналізації статті.
    """

    enabled: bool = True
    patience: int = 10          # (a) ітерацій без покращення best L
    min_delta: float = 1e-4     # (a) мін. абсолютне зменшення L, що рахується покращенням
    window: int = 10            # (b) W — вікно усереднення |ΔL|
    rel_factor: float = 50.0    # (b) X — стоп, коли recent_rate < init_rate / X
    warmup: int = 20            # мін. ітерацій перед будь-якою зупинкою (має бути ≥ window)
    rule: StopRule = "any"


class _EarlyStopper:
    """Стан критерію ранньої зупинки. `update(L_t)` → причина ('patience'|'rel'|...) або None."""

    def __init__(self, cfg: EarlyStopConfig) -> None:
        self.cfg = cfg
        self.best = float("inf")
        self.no_improve = 0
        self._prev: Optional[float] = None
        self._init_deltas: List[float] = []     # |ΔL| перших `window` кроків
        self._init_rate: Optional[float] = None
        self._recent: deque = deque(maxlen=cfg.window)
        self.seen = 0

    def update(self, loss: float) -> Optional[str]:
        cfg = self.cfg
        self.seen += 1

        # (a) patience по best
        if loss < self.best - cfg.min_delta:
            self.best = loss
            self.no_improve = 0
        else:
            self.no_improve += 1

        # (b) швидкість зміни втрати: накопичуємо |ΔL| у перше вікно (для r_init) та ковзне (для r_t)
        if self._prev is not None:
            d = abs(loss - self._prev)
            self._recent.append(d)
            if len(self._init_deltas) < cfg.window:
                self._init_deltas.append(d)
                if len(self._init_deltas) == cfg.window:
                    self._init_rate = sum(self._init_deltas) / cfg.window
        self._prev = loss

        if not cfg.enabled or self.seen < cfg.warmup:
            return None

        cond_a = self.no_improve >= cfg.patience
        cond_b = (
            self._init_rate is not None
            and self._init_rate > 0.0
            and len(self._recent) == cfg.window
            and (sum(self._recent) / cfg.window) < self._init_rate / cfg.rel_factor
        )

        if cfg.rule == "patience":
            fired = cond_a
        elif cfg.rule == "rel":
            fired = cond_b
        elif cfg.rule == "all":
            fired = cond_a and cond_b
        else:  # "any"
            fired = cond_a or cond_b
        if not fired:
            return None
        if cond_a and cond_b:
            return "patience+rel"
        return "patience" if cond_a else "rel"


def _class_loss(
    logits: torch.Tensor, target_logits: torch.Tensor, mode: LossMode, target_class: int
) -> torch.Tensor:
    # cme-bench: для K=2 гілки 'logits' і 'margin' ТОТОЖНІ оригіналу. Для K>2 (ImageNetV2):
    #  * logits — MSE множиться на K/2, щоб вага класового доданка не падала в K/2 разів;
    #  * margin — маржа target проти найсильнішого ІНШОГО класу (для K=2 = старий margin);
    #  * target — зберігається лише логіт target_class (клас-специфічний варіант, sanity checks).
    k = logits.shape[1]
    if mode == "logits":
        loss = F.mse_loss(logits, target_logits)
        return loss if k == 2 else loss * (k / 2.0)
    if mode == "margin":
        if k == 2:
            other = 1 - target_class
            margin_t = target_logits[:, target_class] - target_logits[:, other]
            margin = logits[:, target_class] - logits[:, other]
            return (margin - margin_t).square().mean()
        others = torch.ones(k, dtype=torch.bool, device=logits.device)
        others[target_class] = False
        margin_t = target_logits[:, target_class] - target_logits[:, others].max(dim=1).values
        margin = logits[:, target_class] - logits[:, others].max(dim=1).values
        return (margin - margin_t).square().mean()
    if mode == "target":
        return (logits[:, target_class] - target_logits[:, target_class]).square().mean()
    raise ValueError(f"Unknown loss_mode: {mode!r}")


def optimize_alpha_ext(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    config: DegradationConfig,
    loss_mode: LossMode = "logits",
    early_stop: Optional[EarlyStopConfig] = None,
    base: Optional[torch.Tensor] = None,
    target_class: Optional[int] = None,
    snapshot_steps: Optional[List[int]] = None,
) -> Dict[str, object]:
    """CME-оптимізація з вибором варіанта класової втрати. Повертає alpha + логіти + цільовий клас.

    ВАЖЛИВО: для loss_mode='logits' ця функція алгоритмічно ІДЕНТИЧНА оригінальній
    vit_degradation.optimize_alpha (той самий sigmoid-параметр, той самий mixed, та сама
    регуляризація). Відмінність у результаті йде лише від cfg / loss_mode / зображення, а не
    від самої функції.
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    # Ціль збереження: логіти оригінального (недеградованого) зображення.
    # target_class потрібен лише для loss_mode='margin' (для 'logits' не використовується).
    with torch.no_grad():
        target_logits = model(image_tensor).detach()
        if target_class is None:
            target_class = int(target_logits.argmax(dim=1).item())

    # Фон B — СЕРЕДНІЙ КОЛІР зображення (канонічний baseline CME; doc/comparison_protocol.md §5.1).
    # У нього "тонуть" незначимі зони. (Нульовий/чорний фон base=torch.zeros_like — лишається як
    # експеримент впливу фону, але НЕ канонічний; пов'язаний з bias-аналізом суцільного кольору.)
    # cme-bench: `base` можна задати явно (фони mean/black/noise/blur, у просторі МОДЕЛІ);
    # None → середній колір (канон статті). `snapshot_steps` → alpha+логіти після кроків t (залежність від T).
    if base is None:
        base = image_tensor.mean(dim=(2, 3), keepdim=True).expand_as(image_tensor)
    else:
        base = base.to(device).expand_as(image_tensor)
    snaps = set(snapshot_steps or [])
    snapshots: Dict[int, Dict[str, torch.Tensor]] = {}

    # Оптимізуємо НЕ саму alpha, а її логіт: alpha = sigmoid(alpha_logits) ∈ (0,1) без clip.
    # init_alpha_logit=4.0 → стартова alpha ≈ 0.982 (майже все зображення збережене).
    alpha_logits = torch.nn.Parameter(
        torch.full_like(image_tensor, fill_value=float(config.init_alpha_logit), device=device)
    )
    optimizer = torch.optim.Adam([alpha_logits], lr=config.lr) # Всі значення alpha_logits на початку однакові (config.init_alpha_logit).
    history = {'step': [], 'class_loss': [], 'smooth_loss': [], 'energy_loss': [], 'total_loss': []} # Для збереження проміжних результатів (необов'язково, для аналізу).

    # Рання зупинка за плато втрати (поверх жорсткої межі config.iterations). None → лише max_iter.
    stopper = _EarlyStopper(early_stop) if early_stop is not None else None
    stop_reason: Optional[str] = None
    iterations_run = config.iterations

    for step in range(config.iterations):
        alpha = torch.sigmoid(alpha_logits)                          # маска прозорості ∈ (0,1)
        mixed = base * (1.0 - alpha) + image_tensor * alpha          # синтез: фон vs оригінал

        logits = model(mixed)                                        # прогноз на деградованому

        # 1) Класова втрата — зберегти рішення (logits) або лише маржу класу (margin).
        class_loss = _class_loss(logits, target_logits, loss_mode, target_class)
        # 2) Гладкість маски (Sobel) — карає різкі переходи alpha; БІЛЬША вага → менше "крупинок".
        smooth_loss = _smoothness_loss(alpha, config.err_method)
        # 3) Енергія маски E[alpha^2] — карає великі значення alpha; БІЛЬША вага → більша частина оригінального зображення збережена. Якщо все зображення збережене (alpha≈1), то енергія ≈ 1.0; якщо маска знищує зображення повністю (alpha≈0), то енергія ≈ 0.0.
        energy_loss = alpha.square().mean()

        # Регуляризатор: weight_sharpness балансує гладкість vs енергію всередині weight_blur.
        # Енергетична вага стискання = weight_blur * (1 - weight_sharpness).
        # Якщо сильно стискати (велика енергія) при малій гладкості — маска стає спекловою/розрідженою.
        reg = config.weight_sharpness * smooth_loss + (1.0 - config.weight_sharpness) * energy_loss
        total_loss = config.weight_class * class_loss + config.weight_blur * reg

        history['step'].append(step)
        history['class_loss'].append(class_loss.item())
        history['smooth_loss'].append(smooth_loss.item())
        history['energy_loss'].append(energy_loss.item())
        history['total_loss'].append(total_loss.item())

        optimizer.zero_grad(set_to_none=True) # zero_grad потрібний для занулення градієнтів перед backward. set_to_none=True може бути ефективнішим, але вимагає обережності (не всі оптимізатори підтримують).
        total_loss.backward() # Обчислення градієнтів для alpha_logits.
        optimizer.step() # Оновлення alpha_logits на основі градієнтів.
        if (step + 1) in snaps:
            with torch.no_grad():
                a_t = torch.sigmoid(alpha_logits)
                snapshots[step + 1] = {"alpha": a_t.detach().cpu(),
                                       "logits": model(base * (1.0 - a_t) + image_tensor * a_t).detach().cpu()}

        if stopper is not None:
            stop_reason = stopper.update(history['total_loss'][-1])
            if stop_reason is not None:
                iterations_run = step + 1
                break

    # Фінальна alpha та логіти деградованого зображення (для звітності/перевірки збереження класу).
    with torch.no_grad():
        alpha = torch.sigmoid(alpha_logits)
        mixed = base * (1.0 - alpha) + image_tensor * alpha
        final_logits = model(mixed)

    return {
        "alpha": alpha.detach().cpu(),
        "final_logits": final_logits.detach().cpu(),
        "target_logits": target_logits.detach().cpu(),
        "target_class": target_class,
        "history": history,
        "iterations_run": iterations_run,
        "stop_reason": stop_reason,
        "snapshots": snapshots,
    }

def optimize_alpha_base(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    config: DegradationConfig,
    loss_mode: LossMode = "logits",
    early_stop: Optional[EarlyStopConfig] = None,
) -> Dict[str, object]:
    """CME-оптимізація з вибором варіанта класової втрати. Повертає alpha + логіти + цільовий клас.

    ВАЖЛИВО: для loss_mode='logits' ця функція алгоритмічно ІДЕНТИЧНА оригінальній
    vit_degradation.optimize_alpha (той самий sigmoid-параметр, той самий mixed, та сама
    регуляризація). Відмінність у результаті йде лише від cfg / loss_mode / зображення, а не
    від самої функції.
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    # Ціль збереження: логіти оригінального (недеградованого) зображення.
    # target_class потрібен лише для loss_mode='margin' (для 'logits' не використовується).
    with torch.no_grad():
        target_logits = model(image_tensor).detach()
        target_class = int(target_logits.argmax(dim=1).item())

    # Фон B — середній колір зображення (як в оригінальному CME). У нього "тонуть" незначимі зони.
    base = image_tensor.mean(dim=(2, 3), keepdim=True).expand_as(image_tensor)

    # Оптимізуємо НЕ саму alpha, а її логіт: alpha = sigmoid(alpha_logits) ∈ (0,1) без clip.
    # init_alpha_logit=4.0 → стартова alpha ≈ 0.982 (майже все зображення збережене).
    alpha = torch.nn.Parameter(
        torch.full_like(image_tensor, fill_value=float(config.init_alpha_logit), device=device).clamp(0.0, 1.0)
    )
    optimizer = torch.optim.Adam([alpha], lr=config.lr) # Всі значення alpha на початку однакові (config.init_alpha_logit, затиснуто в [0,1]).
    history = {'step': [], 'class_loss': [], 'smooth_loss': [], 'energy_loss': [], 'total_loss': []}

    # Рання зупинка за плато втрати (поверх жорсткої межі config.iterations). None → лише max_iter.
    stopper = _EarlyStopper(early_stop) if early_stop is not None else None
    stop_reason: Optional[str] = None
    iterations_run = config.iterations

    for step in range(config.iterations):
        alpha.data.clamp_(0.0, 1.0) # In-place затискання: не руйнує leaf-статус Parameter.
        mixed = base * (1.0 - alpha) + image_tensor * alpha          # синтез: фон vs оригінал

        logits = model(mixed)                                        # прогноз на деградованому

        # 1) Класова втрата — зберегти рішення (logits) або лише маржу класу (margin).
        class_loss = _class_loss(logits, target_logits, loss_mode, target_class)
        # 2) Гладкість маски (Sobel) — карає різкі переходи alpha; БІЛЬША вага → менше "крупинок".
        smooth_loss = _smoothness_loss(alpha, config.err_method)
        # 3) Енергія маски E[alpha^2] — карає великі значення alpha; БІЛЬША вага → більша частина оригінального зображення збережена. Якщо все зображення збережене (alpha≈1), то енергія ≈ 1.0; якщо маска знищує зображення повністю (alpha≈0), то енергія ≈ 0.0.
        energy_loss = alpha.square().mean()

        # Регуляризатор: weight_sharpness балансує гладкість vs енергію всередині weight_blur.
        # Енергетична вага стискання = weight_blur * (1 - weight_sharpness).
        # Якщо сильно стискати (велика енергія) при малій гладкості — маска стає спекловою/розрідженою.
        reg = config.weight_sharpness * smooth_loss + (1.0 - config.weight_sharpness) * energy_loss
        total_loss = config.weight_class * class_loss + config.weight_blur * reg

        history['step'].append(step)
        history['class_loss'].append(class_loss.item())
        history['smooth_loss'].append(smooth_loss.item())
        history['energy_loss'].append(energy_loss.item())
        history['total_loss'].append(total_loss.item())

        optimizer.zero_grad(set_to_none=True) # zero_grad потрібний для занулення градієнтів перед backward. set_to_none=True може бути ефективнішим, але вимагає обережності (не всі оптимізатори підтримують).
        total_loss.backward() # Обчислення градієнтів для alpha_logits.
        optimizer.step() # Оновлення alpha_logits на основі градієнтів.

        if stopper is not None:
            stop_reason = stopper.update(history['total_loss'][-1])
            if stop_reason is not None:
                iterations_run = step + 1
                break

    # Фінальна alpha та логіти деградованого зображення (для звітності/перевірки збереження класу).
    with torch.no_grad():
        mixed = base * (1.0 - alpha) + image_tensor * alpha
        final_logits = model(mixed)

    return {
        "alpha": alpha.detach().cpu(),
        "final_logits": final_logits.detach().cpu(),
        "target_logits": target_logits.detach().cpu(),
        "target_class": target_class,
        "history": history,
        "iterations_run": iterations_run,
        "stop_reason": stop_reason,
    }


def cgd_importance(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    config: DegradationConfig,
    loss_mode: LossMode = CGD_LOSS_MODE,
    max_area: float = CGD_MAX_AREA,
    base_variant: bool = False,
    early_stop: Optional[EarlyStopConfig] = None,
    base: Optional[torch.Tensor] = None,
    target_class: Optional[int] = None,
    snapshot_steps: Optional[List[int]] = None,
):
    """Run CME and return (alpha_2d [H,W] float32, cgd_mask bool [H,W], area_frac, raw_result).

    `early_stop=None` → лише жорстка межа config.iterations (відтворює наявні npz). Передайте
    EarlyStopConfig(...), щоб увімкнути зупинку за плато втрати; res додатково містить
    'history', 'iterations_run', 'stop_reason'.
    """
    res = None
    if base_variant:
        res = optimize_alpha_base(model, image_tensor, device, config, loss_mode, early_stop=early_stop)
    else:
        res = optimize_alpha_ext(model, image_tensor, device, config, loss_mode, early_stop=early_stop,
                                 base=base, target_class=target_class, snapshot_steps=snapshot_steps)
    alpha_2d = res["alpha"].mean(dim=1).squeeze(0).numpy().astype(np.float32)
    mask, _ = binarize_alpha_2d(alpha_2d, max_area)
    return alpha_2d, mask, float(mask.mean()), res
