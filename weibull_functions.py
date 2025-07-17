from math import exp, pow

# Weibull-функции без сторонних библиотек
def pdf_weibull(t, k, lam):
    """
    Плотность вероятности отказа f(t)
    """
    return (k / lam) * (t / lam) ** (k - 1) * exp(-(t / lam) ** k)

def cdf_weibull(t, k, lam):
    """
    Функция распределения отказов F(t)
    """
    return 1 - exp(-(t / lam) ** k)

def reliability_weibull(t, k, lam):
    """
    Функция надежности R(t) = 1 - F(t)
    """
    return exp(-(t / lam) ** k)

def hazard_weibull(t, k, lam):
    """
    Интенсивность отказов h(t) = f(t) / R(t)
    """
    f = pdf_weibull(t, k, lam)
    R = reliability_weibull(t, k, lam)
    return f / R

def weibull_pdf_max_point(k, lam):
    """
    Максимум плотности (мода)
    """
    if k <= 1:
        return 0.0
    return lam * pow((k - 1) / k, 1 / k)
