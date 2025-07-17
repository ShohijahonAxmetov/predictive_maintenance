import weibull_functions
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