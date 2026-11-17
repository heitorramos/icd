from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "slides/assets/aula23-regularizacao"
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(ROOT / "exemplos/16-correlacao/data/insurance.csv")
sns.set_theme(style="whitegrid", context="talk")
BLUE, ORANGE, GREEN, PURPLE = "#1f6f8b", "#d95f02", "#2a9d8f", "#6a3d9a"
rng = np.random.default_rng(20260827)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def standardize_train_test(Xtr, Xte):
    mean, scale = Xtr.mean(0), Xtr.std(0)
    scale[scale == 0] = 1
    return (Xtr-mean)/scale, (Xte-mean)/scale


def ridge(X, y, alpha):
    return np.linalg.solve(X.T@X + alpha*np.eye(X.shape[1]), X.T@y)


def soft_threshold(value, threshold):
    return np.sign(value) * max(abs(value) - threshold, 0.0)


def lasso_coordinate_descent(X, y, alpha, max_iter=20_000, tol=1e-9):
    """Minimiza ||y-Xb||²/(2n) + alpha*||b||_1 para X padronizada."""
    n, p = X.shape
    beta = np.zeros(p)
    column_scale = np.sum(X**2, axis=0) / n
    for _ in range(max_iter):
        previous = beta.copy()
        for j in range(p):
            partial_residual = y - X @ beta + X[:, j] * beta[j]
            score = X[:, j] @ partial_residual / n
            beta[j] = soft_threshold(score, alpha) / column_scale[j]
        if np.max(np.abs(beta - previous)) < tol:
            break
    return beta


age = df.age.to_numpy(float)
y = df.charges.to_numpy(float)
smoker = (df.smoker == "yes").to_numpy(float)
bmi = df.bmi.to_numpy(float)
idx = rng.permutation(len(df)); cut = int(.75*len(df)); tr, te = idx[:cut], idx[cut:]

# Data overview
fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
sns.scatterplot(data=df, x="age", y="charges", hue="smoker", alpha=.45, ax=axes[0])
axes[0].set_title("Despesas dependem de vários fatores")
sns.scatterplot(data=df, x="bmi", y="charges", hue="smoker", alpha=.45, ax=axes[1], legend=False)
axes[1].set_title("Há grupos e relações não lineares")
save(fig, "base-descritiva.png")

# Polynomial overfit illustration
x = np.linspace(-1, 1, 24); truth = 5 + 3*x + 2*x**2
y_small = truth + rng.normal(0, 1.2, len(x)); grid = np.linspace(-1.08, 1.08, 400)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
for ax, degree, color in zip(axes, [1, 2, 15], [ORANGE, GREEN, PURPLE]):
    coef = np.polyfit(x, y_small, degree); pred = np.polyval(coef, grid)
    ax.scatter(x, y_small, s=25, alpha=.7, color=BLUE); ax.plot(grid, pred, lw=3, color=color)
    ax.plot(grid, 5+3*grid+2*grid**2, "--", lw=2, color="black", alpha=.6)
    ax.set(title=f"grau {degree}", xlabel="x", ylim=(0, 14))
axes[0].set_ylabel("y")
save(fig, "complexidade-polinomial.png")

# Bias variance simulated
grid = np.linspace(-1, 1, 160); truth_grid = 5+3*grid+2*grid**2
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
regimes = [
    "alto viés\nbaixa variância",
    "baixo viés\nbaixa variância",
    "baixo viés\nalta variância",
]
for ax, degree, color, regime in zip(axes, [1, 2, 12], [ORANGE, GREEN, PURPLE], regimes):
    predictions=[]
    for _ in range(70):
        yy=truth+rng.normal(0,1.2,len(x)); predictions.append(np.polyval(np.polyfit(x,yy,degree),grid))
    predictions=np.array(predictions)
    for p in predictions[:18]: ax.plot(grid,p,color=color,alpha=.13,lw=1)
    ax.plot(grid,predictions.mean(0),color=color,lw=3,label="média dos ajustes")
    ax.plot(grid,truth_grid,"--",color="black",lw=2,label="função real")
    ax.set(xlabel="x", ylim=(0,14))
    ax.set_title(f"grau {degree}\n{regime}", color=color, fontsize=15, fontweight="bold")
axes[0].set_ylabel("y"); axes[-1].legend(fontsize=10)
save(fig,"vies-variancia.png")

# Bias-variance trade-off and the three fitting regimes
complexity = np.linspace(0, 1, 500)
bias2 = 0.82 * np.exp(-4.2 * complexity) + 0.025
variance = 0.035 + 0.72 * complexity**3
noise = np.full_like(complexity, 0.11)
generalization_error = bias2 + variance + noise
best = int(np.argmin(generalization_error))
best_complexity = complexity[best]
left_limit = best_complexity - 0.17
right_limit = best_complexity + 0.17

fig, ax = plt.subplots(figsize=(12.5, 5.5))
ax.axvspan(0, left_limit, color=ORANGE, alpha=0.08)
ax.axvspan(left_limit, right_limit, color=GREEN, alpha=0.10)
ax.axvspan(right_limit, 1, color=PURPLE, alpha=0.07)
ax.plot(complexity, bias2, color=ORANGE, lw=3, label=r"viés$^2$")
ax.plot(complexity, variance, color=PURPLE, lw=3, label="variância")
ax.plot(
    complexity,
    generalization_error,
    color=BLUE,
    lw=4,
    label="erro esperado fora da amostra",
)
ax.axhline(noise[0], color="#66727c", lw=2, ls="--", label="ruído irredutível")
ax.axvline(best_complexity, color=GREEN, lw=2.2, ls="--")
ax.scatter(
    [best_complexity],
    [generalization_error[best]],
    s=105,
    color=GREEN,
    edgecolor="white",
    linewidth=1.5,
    zorder=5,
)
ax.annotate(
    "menor erro esperado",
    xy=(best_complexity, generalization_error[best]),
    xytext=(best_complexity + 0.04, generalization_error[best] + 0.13),
    arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1.8},
    color=GREEN,
    fontsize=13,
    fontweight="bold",
)
ax.text(left_limit / 2, 0.96, "subajuste", ha="center", va="top", color=ORANGE, fontweight="bold")
ax.text((left_limit + right_limit) / 2, 0.96, "ajuste adequado", ha="center", va="top", color=GREEN, fontweight="bold")
ax.text((right_limit + 1) / 2, 0.96, "sobreajuste", ha="center", va="top", color=PURPLE, fontweight="bold")
ax.set(
    xlim=(0, 1),
    ylim=(0, 1.01),
    xlabel="complexidade do modelo",
    ylabel="erro esperado",
)
ax.set_xticks([])
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False, fontsize=11)
save(fig, "subajuste-ajuste-sobreajuste.png")

# Insurance example: the same polynomial basis with and without regularization
insurance_subset = df.loc[df["smoker"] == "no", ["age", "charges"]].copy()
insurance_subset = insurance_subset.iloc[rng.choice(len(insurance_subset), 150, replace=False)]
insurance_subset = insurance_subset.sample(frac=1, random_state=20261117).reset_index(drop=True)
train_insurance = insurance_subset.iloc[:55]
valid_insurance = insurance_subset.iloc[55:]
age_mean = train_insurance["age"].mean()
age_scale = train_insurance["age"].std()


def polynomial_basis(values, degree=12):
    standardized = (np.asarray(values) - age_mean) / age_scale
    return np.vander(standardized, N=degree + 1, increasing=True)


X_poly_train_raw = polynomial_basis(train_insurance["age"])
poly_mean = X_poly_train_raw[:, 1:].mean(axis=0)
poly_scale = X_poly_train_raw[:, 1:].std(axis=0)
poly_scale[poly_scale == 0] = 1


def polynomial_design(values, degree=12):
    design = polynomial_basis(values, degree=degree)
    design[:, 1:] = (design[:, 1:] - poly_mean) / poly_scale
    return design


X_poly_train = polynomial_design(train_insurance["age"])
X_poly_valid = polynomial_design(valid_insurance["age"])
y_poly_train = train_insurance["charges"].to_numpy()
y_poly_valid = valid_insurance["charges"].to_numpy()
beta_ols_poly = np.linalg.lstsq(X_poly_train, y_poly_train, rcond=None)[0]
penalty_poly = np.eye(X_poly_train.shape[1])
penalty_poly[0, 0] = 0
ridge_grid = np.logspace(-4, 5, 160)
ridge_betas = np.array([
    np.linalg.solve(X_poly_train.T @ X_poly_train + lam * penalty_poly, X_poly_train.T @ y_poly_train)
    for lam in ridge_grid
])
ridge_valid_rmse = np.array([
    np.sqrt(np.mean((y_poly_valid - X_poly_valid @ beta) ** 2))
    for beta in ridge_betas
])
best_ridge = int(np.argmin(ridge_valid_rmse))
beta_ridge_poly = ridge_betas[best_ridge]
ols_valid_rmse = np.sqrt(np.mean((y_poly_valid - X_poly_valid @ beta_ols_poly) ** 2))
age_grid = np.linspace(18, 64, 400)
X_poly_grid = polynomial_design(age_grid)
ols_curve = X_poly_grid @ beta_ols_poly
ridge_curve = X_poly_grid @ beta_ridge_poly

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True, sharey=True)
for ax in axes:
    ax.scatter(
        train_insurance["age"],
        train_insurance["charges"],
        color=BLUE,
        alpha=0.58,
        s=35,
        label="treino",
    )
    ax.scatter(
        valid_insurance["age"],
        valid_insurance["charges"],
        color="#9aa8b3",
        alpha=0.28,
        s=24,
        label="validação",
    )
    ax.set(xlabel="idade", xlim=(17, 65), ylim=(0, 26000))
axes[0].plot(age_grid, ols_curve, color=ORANGE, lw=3)
axes[0].set_title("Sem penalização: curva muito flexível")
axes[0].text(
    0.04,
    0.92,
    "RMSE validação = " + f"{ols_valid_rmse:,.0f}".replace(",", "."),
    transform=axes[0].transAxes,
    color=ORANGE,
    fontweight="bold",
)
axes[1].plot(age_grid, ridge_curve, color=GREEN, lw=3)
axes[1].set_title("Ridge: mesmos termos, coeficientes contraídos")
axes[1].text(
    0.04,
    0.92,
    "RMSE validação = " + f"{ridge_valid_rmse[best_ridge]:,.0f}".replace(",", "."),
    transform=axes[1].transAxes,
    color=GREEN,
    fontweight="bold",
)
axes[0].set_ylabel("despesas médicas anuais")
axes[1].legend(loc="lower right", fontsize=10)
save(fig, "insurance-flexibilidade-regularizacao.png")

# Same nonlinear model under several L2 penalties
panel_lambdas = [0, 0.1, 1, 16, 100, 10000]
fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), sharex=True, sharey=True)
for ax, lam in zip(axes.flat, panel_lambdas):
    if lam == 0:
        beta_panel = beta_ols_poly
    else:
        beta_panel = np.linalg.solve(
            X_poly_train.T @ X_poly_train + lam * penalty_poly,
            X_poly_train.T @ y_poly_train,
        )
    validation_rmse = np.sqrt(np.mean((y_poly_valid - X_poly_valid @ beta_panel) ** 2))
    curve = X_poly_grid @ beta_panel
    ax.scatter(
        train_insurance["age"],
        train_insurance["charges"],
        color=BLUE,
        alpha=0.52,
        s=22,
    )
    ax.scatter(
        valid_insurance["age"],
        valid_insurance["charges"],
        color="#9aa8b3",
        alpha=0.22,
        s=16,
    )
    ax.plot(age_grid, curve, color=GREEN if lam else ORANGE, lw=2.6)
    if lam == 0:
        lambda_label = "0"
    elif lam < 1:
        lambda_label = f"{lam:.1f}".replace(".", ",")
    else:
        lambda_label = f"{lam:,.0f}".replace(",", ".")
    rmse_label = f"{validation_rmse:,.0f}".replace(",", ".")
    ax.set_title(fr"$\lambda={lambda_label}$   |   RMSE = {rmse_label}", fontsize=13)
    ax.set(xlim=(17, 65), ylim=(0, 26000))
for ax in axes[-1]:
    ax.set_xlabel("idade")
for ax in axes[:, 0]:
    ax.set_ylabel("despesas anuais")
fig.suptitle("Mesmo polinômio de grau 12, diferentes intensidades de regularização L2", fontsize=18, fontweight="bold")
save(fig, "insurance-varios-lambdas.png")

# Rich feature matrix for insurance
raw = np.column_stack([age, bmi, smoker, age*bmi, age*smoker, bmi*smoker,
                       age**2, bmi**2, age**3, bmi**3])
names = ["age", "bmi", "smoker", "age:bmi", "age:smoker", "bmi:smoker",
         "age²", "bmi²", "age³", "bmi³"]
Xtr, Xte = standardize_train_test(raw[tr], raw[te]); ymean=y[tr].mean(); yc=y[tr]-ymean
alphas=np.logspace(-3,5,120); coefs=np.array([ridge(Xtr,yc,a) for a in alphas])
rmse_tr=np.array([np.sqrt(np.mean((yc-Xtr@b)**2)) for b in coefs])
rmse_te=np.array([np.sqrt(np.mean((y[te]-(ymean+Xte@b))**2)) for b in coefs])

fig, ax=plt.subplots(figsize=(9,5.2))
for j,n in enumerate(names): ax.semilogx(alphas,coefs[:,j],lw=2,label=n)
ax.axhline(0,color="black",lw=1); ax.set(xlabel=r"penalização $\lambda$",ylabel="coeficiente",title="Ridge contrai coeficientes continuamente")
ax.legend(ncol=2,fontsize=9)
save(fig,"caminho-ridge.png")

fig,ax=plt.subplots(figsize=(9,5.2)); ax.semilogx(alphas,rmse_tr,lw=3,label="treino",color=BLUE); ax.semilogx(alphas,rmse_te,lw=3,label="validação",color=ORANGE)
best=np.argmin(rmse_te); ax.axvline(alphas[best],ls="--",color="black",label=fr"melhor $\lambda$={alphas[best]:.2g}")
ax.set(xlabel=r"penalização $\lambda$",ylabel="RMSE",title="Validação escolhe a complexidade"); ax.legend()
save(fig,"curva-validacao.png")

# Geometry of L0, L1 and L2 constraints against the same objective
u = np.linspace(-3.0, 3.6, 520)
U, V = np.meshgrid(u, u)
# O mínimo sem restrição e a orientação das elipses foram escolhidos para que
# a primeira curva de nível alcance a bola L1 exatamente no vértice (2, 0).
# Assim, o exemplo torna visível a seleção de atributos: beta_2 é zerado.
center = np.array([2.55, 0.55])
quadratic = np.array([[1.20, 0.15], [0.15, 0.85]])
DU, DV = U - center[0], V - center[1]
loss = quadratic[0, 0] * DU**2 + 2 * quadratic[0, 1] * DU * DV + quadratic[1, 1] * DV**2


def quadratic_value(points):
    differences = np.asarray(points) - center
    return np.einsum("...i,ij,...j->...", differences, quadratic, differences)


axis_values = np.linspace(-3.0, 3.6, 3000)
l0_candidates = np.vstack([
    np.column_stack([axis_values, np.zeros_like(axis_values)]),
    np.column_stack([np.zeros_like(axis_values), axis_values]),
])
l1_radius = 2.0
l1_x = np.linspace(-l1_radius, l1_radius, 2400)
l1_candidates = np.vstack([
    np.column_stack([l1_x, l1_radius - np.abs(l1_x)]),
    np.column_stack([l1_x, -l1_radius + np.abs(l1_x)]),
])
l2_radius = 1.85
angles = np.linspace(0, 2*np.pi, 3000)
l2_candidates = np.column_stack([l2_radius*np.cos(angles), l2_radius*np.sin(angles)])

constraint_candidates = [l0_candidates, l1_candidates, l2_candidates]
selected_points = [points[np.argmin(quadratic_value(points))] for points in constraint_candidates]
constraint_colors = [BLUE, ORANGE, GREEN]
constraint_titles = [
    r"$L_0$: $\|\beta\|_0\leq 1$",
    r"$L_1$: $\|\beta\|_1\leq t$",
    r"$L_2$: $\|\beta\|_2\leq t$",
]

fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8))
constraint_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
for ax, kind, color, selected in zip(
    constraint_axes, ["L0", "L1", "L2"], constraint_colors, selected_points
):
    touching_level = float(quadratic_value(selected))
    contour_levels = np.array([0.20, 0.45, 0.72, 1.0, 1.45, 2.1]) * touching_level
    ax.contour(U, V, loss, levels=contour_levels, colors="#9fb4c2", linewidths=1.2)
    ax.contour(U, V, loss, levels=[touching_level], colors="#263746", linewidths=2.6)

    if kind == "L0":
        ax.plot([-3, 3.6], [0, 0], color=color, lw=5, alpha=.72)
        ax.plot([0, 0], [-3, 3.6], color=color, lw=5, alpha=.72)
        ax.text(-2.72, -2.55, "conjunto não convexo", color=color, fontsize=10, fontweight="bold")
    elif kind == "L1":
        diamond = np.array([[0, l1_radius], [l1_radius, 0], [0, -l1_radius], [-l1_radius, 0]])
        ax.fill(diamond[:, 0], diamond[:, 1], color=color, alpha=.16)
        ax.plot(np.r_[diamond[:, 0], diamond[0, 0]], np.r_[diamond[:, 1], diamond[0, 1]], color=color, lw=3)
    else:
        ax.fill(l2_candidates[:, 0], l2_candidates[:, 1], color=color, alpha=.16)
        ax.plot(l2_candidates[:, 0], l2_candidates[:, 1], color=color, lw=3)

    ax.scatter(*center, marker="*", s=135, color="#66727c", zorder=7, label="mínimo sem restrição")
    ax.scatter(*selected, s=95, color="#111111", edgecolor="white", linewidth=1.2, zorder=8, label="solução regularizada")
    if kind == "L1":
        ax.annotate(
            r"$\widehat\beta_2=0$",
            xy=selected,
            xytext=(2.58, -0.82),
            color=color,
            fontsize=11,
            fontweight="bold",
            ha="center",
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.8},
        )
    ax.set(
        aspect="equal",
        xlim=(-3, 3.6),
        ylim=(-3, 3.6),
        xlabel=r"$\beta_1$",
        ylabel=r"$\beta_2$",
        title=constraint_titles[["L0", "L1", "L2"].index(kind)],
    )

guide = axes[1, 1]
guide.axis("off")
guide.scatter(0.12, 0.78, marker="*", s=180, color="#66727c", transform=guide.transAxes)
guide.text(0.22, 0.78, "mínimo sem restrição", transform=guide.transAxes, va="center", fontsize=12)
guide.scatter(0.12, 0.62, s=115, color="#111111", edgecolor="white", linewidth=1.2, transform=guide.transAxes)
guide.text(0.22, 0.62, "solução regularizada", transform=guide.transAxes, va="center", fontsize=12)
guide.plot([0.07, 0.17], [0.46, 0.46], color="#263746", lw=3, transform=guide.transAxes)
guide.text(0.22, 0.46, "primeira curva que toca", transform=guide.transAxes, va="center", fontsize=12)
guide.text(
    0.07,
    0.25,
    "A forma da região permitida\ndetermina onde o contato ocorre.",
    transform=guide.transAxes,
    va="center",
    fontsize=12.5,
    color="#263746",
    fontweight="bold",
)
save(fig, "geometria-l0-l1-l2.png")

# Other Lp balls. For p < 1 this is a quasi-norm and the ball is non-convex.
ball_grid = np.linspace(-1.25, 1.25, 520)
BU, BV = np.meshgrid(ball_grid, ball_grid)
lp_cases = [
    (0.5, r"$p=\frac{1}{2}$", "não convexa; favorece zeros"),
    (1.5, r"$p=\frac{3}{2}$", "convexa e suave"),
    (4.0, r"$p=4$", "aproxima uma caixa"),
    (np.inf, r"$p=\infty$", r"limita $\max_j|\beta_j|$"),
]
lp_colors = [PURPLE, "#457b9d", "#e07a5f", "#3d405b"]
fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), sharex=True, sharey=True)
for ax, (p_value, p_label, description), color in zip(axes.flat, lp_cases, lp_colors):
    if np.isinf(p_value):
        lp_value = np.maximum(np.abs(BU), np.abs(BV))
    else:
        lp_value = (np.abs(BU)**p_value + np.abs(BV)**p_value)**(1/p_value)
    ax.contourf(BU, BV, lp_value, levels=[0, 1], colors=[color], alpha=.18)
    ax.contour(BU, BV, lp_value, levels=[1], colors=[color], linewidths=3)
    ax.axhline(0, color="#a8b2b9", lw=.8)
    ax.axvline(0, color="#a8b2b9", lw=.8)
    ax.set(aspect="equal", xlim=(-1.18, 1.18), ylim=(-1.18, 1.18))
    ax.set_title(p_label, color=color, fontweight="bold", fontsize=17)
    ax.text(
        0.5,
        0.045,
        description,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=color,
        fontsize=9.5,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.0},
    )
for ax in axes[-1]:
    ax.set_xlabel(r"$\beta_1$")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$\beta_2$")
save(fig, "familia-lp-bolas.png")

# Synthetic example: nearly redundant predictors under repeated samples
synthetic_rng = np.random.default_rng(20261117)
n_repetitions, sample_size = 300, 60
ridge_strength, lasso_strength = 0.25, 0.35
coeficients = {"OLS": [], "Ridge": [], "Lasso": []}
example_predictors = None

for repetition in range(n_repetitions):
    latent = synthetic_rng.normal(size=sample_size)
    x1 = latent + synthetic_rng.normal(scale=0.03, size=sample_size)
    x2 = latent + synthetic_rng.normal(scale=0.03, size=sample_size)
    X_syn = np.column_stack([x1, x2])
    X_syn = (X_syn - X_syn.mean(axis=0)) / X_syn.std(axis=0)
    y_syn = 6 * latent + synthetic_rng.normal(scale=1.2, size=sample_size)
    y_syn = y_syn - y_syn.mean()

    coeficients["OLS"].append(np.linalg.lstsq(X_syn, y_syn, rcond=None)[0])
    coeficients["Ridge"].append(
        np.linalg.solve(
            X_syn.T @ X_syn / sample_size + ridge_strength * np.eye(2),
            X_syn.T @ y_syn / sample_size,
        )
    )
    coeficients["Lasso"].append(
        lasso_coordinate_descent(X_syn, y_syn, lasso_strength)
    )
    if repetition == 0:
        example_predictors = X_syn.copy()

coeficients = {method: np.asarray(values) for method, values in coeficients.items()}
coefficient_limits = (-14, 20)
method_colors = {"OLS": ORANGE, "Ridge": GREEN, "Lasso": PURPLE}

fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
axes[0, 0].scatter(
    example_predictors[:, 0],
    example_predictors[:, 1],
    color=BLUE,
    alpha=0.58,
    s=34,
)
correlation = np.corrcoef(example_predictors.T)[0, 1]
axes[0, 0].set(
    xlabel=r"preditor $x_1$",
    ylabel=r"preditor $x_2$",
    title=fr"Preditores quase repetidos: $r={correlation:.3f}$",
)

for ax, method in zip([axes[0, 1], axes[1, 0], axes[1, 1]], ["OLS", "Ridge", "Lasso"]):
    values = coeficients[method]
    ax.scatter(
        values[:, 0],
        values[:, 1],
        color=method_colors[method],
        alpha=0.34,
        s=30,
        edgecolor="none",
    )
    ax.axline((0, 6), (6, 0), color="#5f6b73", lw=1.6, ls="--")
    ax.axhline(0, color="#9aa8b3", lw=0.9)
    ax.axvline(0, color="#9aa8b3", lw=0.9)
    ax.set(
        xlim=coefficient_limits,
        ylim=coefficient_limits,
        xlabel=r"coeficiente de $x_1$",
        ylabel=r"coeficiente de $x_2$",
        title=method,
    )
    ax.set_aspect("equal", adjustable="box")

axes[0, 1].text(
    0.04,
    0.94,
    "mesma soma, repartição instável",
    transform=axes[0, 1].transAxes,
    va="top",
    color=ORANGE,
    fontweight="bold",
    fontsize=11,
)
axes[1, 0].text(
    0.04,
    0.94,
    "pesos mais equilibrados",
    transform=axes[1, 0].transAxes,
    va="top",
    color=GREEN,
    fontweight="bold",
    fontsize=11,
)
axes[1, 1].text(
    0.04,
    0.94,
    "um coeficiente frequentemente zera",
    transform=axes[1, 1].transAxes,
    va="top",
    color=PURPLE,
    fontweight="bold",
    fontsize=11,
)
save(fig, "correlacionados-ridge-lasso.png")

# Stability under resampling: OLS vs ridge
ols=[]; rid=[]
for _ in range(500):
    ii=rng.integers(0,len(tr),len(tr)); Xi=Xtr[ii]; yi=yc[ii]
    ols.append(np.linalg.lstsq(Xi,yi,rcond=None)[0]); rid.append(ridge(Xi,yi,alphas[best]))
ols=np.array(ols); rid=np.array(rid)
fig,axes=plt.subplots(1,2,figsize=(12,4.7),sharey=True)
sel=[0,1,3,6]
axes[0].boxplot([ols[:,j] for j in sel],labels=[names[j] for j in sel]); axes[0].set_title("Sem regularização")
axes[1].boxplot([rid[:,j] for j in sel],labels=[names[j] for j in sel]); axes[1].set_title("Ridge")
for ax in axes: ax.tick_params(axis="x",rotation=20); ax.axhline(0,color="black",lw=.8)
axes[0].set_ylabel("coeficiente em reamostras")
save(fig,"estabilidade-coeficientes.png")

print({
    "lambda_best": float(alphas[best]),
    "rmse_train": float(rmse_tr[best]),
    "rmse_valid": float(rmse_te[best]),
    "lambda_insurance_polynomial": float(ridge_grid[best_ridge]),
    "rmse_insurance_polynomial": float(ridge_valid_rmse[best_ridge]),
    "synthetic_sd_ols": coeficients["OLS"].std(axis=0).round(3).tolist(),
    "synthetic_sd_ridge": coeficients["Ridge"].std(axis=0).round(3).tolist(),
    "synthetic_lasso_zero_rate": np.mean(np.isclose(coeficients["Lasso"], 0), axis=0).round(3).tolist(),
})
