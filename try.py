import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def model(t, y, a, e1, K, c, e2, k, N0, r, lam1, w1, w2, s1, u, b, s2, d, mu, alpha,gamma):
    N, P1, P2, Z = y
    # 避免除以零
    N_e1 = N + e1 if N + e1 != 0 else 1e-10
    N_e2 = N + e2 if N + e2 != 0 else 1e-10
    P1_lam1 = P1 + lam1 if P1 + lam1 != 0 else 1e-10

    dNdt = (- alpha * N / N_e1 * (1 - P1 / K) * P1
            - (c * N / N_e2) * P2
            + k * (N0 - N))

    dP1dt = (r * N / N_e1 * (1 - P1 / K) * P1
             - (a / P1_lam1) * P1 * Z
             + w2 * P2
             - (k + s1) * P1)

    dP2dt = (u * N / N_e2 * P2
             - b * P2 * Z
             - w2 * P2
             - (k+s2) * P2)

    dZdt = ((a * gamma * P1 / P1_lam1 + d * P2) * Z
            - mu * Z)

    return [dNdt, dP1dt, dP2dt, dZdt]


# 参数设置
params = {
    'a': 0.5, 'e1': 1.0, 'K': 100.0, 'c': 0.2, 'e2': 1.0,
    'k': 0.1, 'N0': 10.0, 'r': 0.3, 'lam1': 1.0, 'w1': 0.05,
    'w2': 0.02, 's1': 0.1, 'u': 0.2, 'b': 0.1, 's2': 0.1,
    'd': 0.05, 'mu': 0.05, 'alpha':0.2 ,'gamma':0.8# 调整mu以观察不同动态
}

# 初始条件和时间范围
y0 = [50, 1, 10, 5]
t_span = (0, 1000)
t_eval = np.linspace(*t_span, 1000)

# 数值求解
sol = solve_ivp(model, t_span, y0, args=tuple(params.values()),
                t_eval=t_eval, method='LSODA')


# 绘图
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='N')
plt.plot(sol.t, sol.y[1], label='P1')
plt.plot(sol.t, sol.y[2], label='P2')
plt.plot(sol.t, sol.y[3], label='Z')
plt.xlabel('Time')
plt.ylabel('Population/Concentration')
plt.legend()
plt.title('Dynamics of Nutrient-Algae-Predator System')
plt.show()