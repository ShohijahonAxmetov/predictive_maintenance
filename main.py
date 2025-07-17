# Лучший способ подбора параметров Weibull регрессии: scipy.stats.weibull_min.fit
import pandas as pd
import numpy as np
from scipy.stats import weibull_min
import matplotlib.pyplot as plt
from math import exp, pow

import weibull_functions

data = pd.read_csv("data/pump_time_to_failure.csv")["time_to_failure_hours"]

# Оценка параметров распределения Вейбулла:
# shape = k (форма), scale = λ (масштаб), loc — смещение (часто = 0)
shape, loc, scale = weibull_min.fit(data, floc=0)  # фиксируем loc=0 (обычно логично)

print("Оценённые параметры Вейбулла (MLE):")
print(f"  → Форма (k):   {shape:.4f}")
print(f"  → Масштаб (λ): {scale:.2f} часов")
print(f"  → Максимум плотности (мода): t = {weibull_functions.weibull_pdf_max_point(shape, scale):.2f} ч")


# Построим функции R(t), f(t), h(t)
t_vals = np.linspace(0, 2000, 300)
f_t = weibull_min.pdf(t_vals, c=shape, scale=scale)
F_t = weibull_min.cdf(t_vals, c=shape, scale=scale)

# Визуализация
plt.figure(figsize=(12, 6))
plt.plot(t_vals, f_t, label="f(t) — Плотность отказов", color='blue')
plt.title("Функции надёжности (Weibull), подобранные по данным")
plt.xlabel("Время (часы)")
plt.ylabel("Значение")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Расчет и показ:
#   - Плотность вероятности отказа f(t)
#   - Функция распределения отказов F(t)
#   - Функция надежности R(t) = 1 - F(t)
#   - Интенсивность отказов h(t) = f(t) / R(t)
def print_all(t):
	f_t = weibull_functions.pdf_weibull(t, shape, scale)
	F_t = weibull_functions.cdf_weibull(t, shape, scale)
	R_t = weibull_functions.reliability_weibull(t, shape, scale)
	h_t = weibull_functions.hazard_weibull(t, shape, scale)

	print(f"t = {t} ч")
	print(f"f(t) — плотность вероятности отказа: {f_t:.6f}")
	print(f"F(t) — функция распределения:        {F_t:.6f}")
	print(f"R(t) — функция надёжности:           {R_t:.6f}")
	print(f"h(t) — интенсивность отказов:        {h_t:.6f}")


# print_all(864)
# print_all(700)
# print_all(1000)


from scipy.stats import gaussian_kde

# Ядерная оценка плотности (KDE)
kde = gaussian_kde(data)

# Сетка по времени
x = np.linspace(0, np.max(data), 1000)
y = kde(x)

# Рисуем
plt.figure(figsize=(10, 5))
plt.plot(x, y, color='royalblue', linewidth=2, label='Оценка плотности отказов')
plt.fill_between(x, y, alpha=0.3, color='skyblue')
plt.title("Функция плотности отказов оборудования", fontsize=14, weight='bold')
plt.xlabel("Время, ч")
plt.ylabel("Плотность отказов")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()









from scipy.optimize import minimize_scalar

# --- Стоимость ---
C_p = 700                      # плановая замена
C_f_repair = 1000             # ремонт при отказе
downtime_hours = 8
cost_per_downtime_hour = 400
production_loss_per_hour = 150
penalty_risk = 300

# --- Общая аварийная стоимость ---
C_f_total = (
    C_f_repair +
    downtime_hours * (cost_per_downtime_hour + production_loss_per_hour) +
    penalty_risk
)

# --- Потери ресурса при преждевременной замене ---
C_amort = C_p  # считаем стоимость недоиспользованного ресурса = стоимости замены

# --- Ожидаемая стоимость ---
def expected_cost(t):
    r = weibull_functions.reliability_weibull(t, shape, scale)
    f = weibull_functions.cdf_weibull(t, shape, scale)
    # Три слагаемых:
    cost_plan = C_p * r
    cost_fail = C_f_total * (1 - r)
    cost_lost_resource = C_amort * r * (1 - f)
    return cost_plan + cost_fail + cost_lost_resource

# --- Удельная стоимость ---
def cost_per_hour(t):
    return expected_cost(t) / t

# --- Удельная стоимость ---
def cost(t):
    return expected_cost(t)

# --- Оптимизация на интервале (например, до 1800 ч) ---
result = minimize_scalar(cost_per_hour, bounds=(100, 1800), method='bounded')
t_opt = result.x
cph_opt = result.fun

# --- График (по желанию) ---
t_vals = np.linspace(100, 1800, 300)
cph_vals = [cost_per_hour(t) for t in t_vals]

# --- Печать результатов ---
print(f"Оптимальное время замены: {t_opt:.0f} ч")
print(f"Минимальные удельные затраты: {cph_opt:.2f} у.е./ч")

# --- Построение графика (если хочешь)
plt.figure(figsize=(10, 6))
plt.plot(t_vals, cph_vals, label='Удельные затраты', color='darkred')
plt.axvline(t_opt, color='blue', linestyle='--', label=f'Оптимум: {t_opt:.0f} ч')
plt.title('Удельные затраты на 1 час эксплуатации')
plt.xlabel('Время замены, ч')
plt.ylabel('у.е./час')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Таблица удельных затрат каждые 200 часов до 2000 ч ---
t_steps = np.arange(200, 2001, 200)

print("\nУдельные затраты на 1 час эксплуатации:")
print("Время (ч)\tУдельные затраты (у.е./ч)")
print("-" * 35)
for t in t_steps:
    cph = cost_per_hour(t)
    print(f"{t:>5} \t\t {cph:.3f}")
