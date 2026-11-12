from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"slides/assets/aula20-gradiente-descendente"
OUT.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(ROOT/"exemplos/16-correlacao/data/insurance.csv")
sns.set_theme(style="whitegrid",context="talk")
blue,orange,green="#1f6f8b","#d95f02","#2a9d8f"
x=df.age.to_numpy(float); y=df.charges.to_numpy(float)
xz=(x-x.mean())/x.std(); yz=(y-y.mean())/y.std()

def save(fig,name):
    fig.tight_layout(); fig.savefig(OUT/name,dpi=180,bbox_inches="tight",facecolor="white"); plt.close(fig)

# One-parameter loss and derivative
beta_star=np.sum(xz*yz)/np.sum(xz*xz)
b=np.linspace(-.8,1.4,500)
loss=np.array([np.mean((yz-v*xz)**2) for v in b])
grad=np.array([2*np.mean(xz*(v*xz-yz)) for v in b])
fig,axes=plt.subplots(1,2,figsize=(12,4.5)); axes[0].plot(b,loss,lw=3,color=blue); axes[0].axvline(beta_star,ls="--",color=orange); axes[0].set(xlabel=r"$\beta$",ylabel="MSE",title="Perda convexa")
axes[1].plot(b,grad,lw=3,color=green); axes[1].axhline(0,color="black"); axes[1].axvline(beta_star,ls="--",color=orange); axes[1].set(xlabel=r"$\beta$",ylabel=r"$dJ/d\beta$",title="O sinal aponta a direção")
save(fig,"perda-derivada.png")

def gd1(lr,steps=25,start=-.7):
    bb=start; hist=[]
    for t in range(steps):
        ll=np.mean((bb*xz-yz)**2); gg=2*np.mean(xz*(bb*xz-yz)); hist.append((t,bb,ll,gg)); bb-=lr*gg
    return np.array(hist)

fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
for ax,lr in zip(axes,[.05,.45,1.05]):
    h=gd1(lr,18); ax.plot(b,loss,color="#bbbbbb",lw=2); ax.plot(h[:,1],h[:,2],"o-",color=orange,lw=2,ms=4); ax.set_title(fr"$\kappa={lr}$"); ax.set_xlabel(r"$\beta$")
axes[0].set_ylabel("MSE"); save(fig,"taxas-aprendizado.png")

# 2D contour and path
def mse(a,bb): return np.mean((yz-a-bb*xz)**2)
aa=np.linspace(-1,1,180); bb=np.linspace(-.5,1.1,180); A,B=np.meshgrid(aa,bb); Z=np.vectorize(mse)(A,B)
theta=np.array([.9,-.35]); path=[]; lr=.22
for t in range(24):
    pred=theta[0]+theta[1]*xz; difference=pred-yz; g=np.array([2*np.mean(difference),2*np.mean(xz*difference)]); path.append(theta.copy()); theta-=lr*g
path=np.array(path)
fig,ax=plt.subplots(figsize=(8,5.5)); cs=ax.contour(A,B,Z,levels=18,cmap="Blues"); ax.clabel(cs,inline=True,fontsize=7); ax.plot(path[:,0],path[:,1],"o-",color=orange,lw=2,ms=4); ax.scatter([0],[beta_star],s=100,color=green,zorder=4); ax.set(xlabel=r"intercepto padronizado $\beta_0$",ylabel=r"inclinação $\beta_1$",title="Gradiente descendente percorre a superfície de perda"); save(fig,"contorno-caminho.png")

# convergence and OLS
h=gd1(.25,50)
fig,axes=plt.subplots(1,2,figsize=(12,4.5)); axes[0].plot(h[:,0],h[:,1],lw=3,color=blue); axes[0].axhline(beta_star,ls="--",color=orange,label="solução OLS"); axes[0].set(xlabel="iteração",ylabel=r"$\beta_t$",title="O parâmetro converge"); axes[0].legend()
axes[1].plot(h[:,0],h[:,2],lw=3,color=green); axes[1].axhline(loss.min(),ls="--",color=orange,label="mínimo OLS"); axes[1].set(xlabel="iteração",ylabel=r"$J(\beta_t)$",title="A perda se estabiliza no mínimo"); axes[1].legend(); save(fig,"convergencia-ols.png")

# SGD vs batch
rng=np.random.default_rng(20260827)
def train(batch,epochs=35,lr=.04):
    th=np.zeros(2); out=[]
    for epoch in range(epochs):
        ids=rng.permutation(len(xz))
        for k in range(0,len(ids),batch):
            ii=ids[k:k+batch]; pred=th[0]+th[1]*xz[ii]; difference=pred-yz[ii]; g=np.array([2*np.mean(difference),2*np.mean(xz[ii]*difference)]); th-=lr*g
        out.append((epoch,mse(*th),*th))
    return np.array(out)
fig,ax=plt.subplots(figsize=(9,5))
for batch,label,c in [(len(xz),"batch completo",blue),(64,"mini-batch 64",green),(1,"SGD",orange)]:
    hh=train(batch); ax.plot(hh[:,0],hh[:,1],lw=2.5,label=label,color=c)
ax.set(xlabel="época",ylabel="MSE",title="Mini-batches trocam suavidade por atualizações baratas"); ax.legend(); save(fig,"batch-sgd.png")

# Parameter paths for batch, mini-batch, and SGD on the same loss surface
def stochastic_path(batch_size, updates=70, learning_rate=.04, seed=20261104):
    local_rng = np.random.default_rng(seed)
    theta = np.array([.9, -.35], dtype=float)
    history = [theta.copy()]
    order = local_rng.permutation(len(xz))
    cursor = 0

    for _ in range(updates):
        if batch_size == len(xz):
            indices = np.arange(len(xz))
        else:
            if cursor + batch_size > len(xz):
                order = local_rng.permutation(len(xz))
                cursor = 0
            indices = order[cursor:cursor+batch_size]
            cursor += batch_size

        prediction = theta[0] + theta[1] * xz[indices]
        difference = prediction - yz[indices]
        gradient = np.array([
            2 * np.mean(difference),
            2 * np.mean(xz[indices] * difference),
        ])
        theta -= learning_rate * gradient
        history.append(theta.copy())

    return np.asarray(history)

method_specs = [
    (len(xz), "Batch completo"),
    (64, "Mini-batch ($m=64$)"),
    (1, "SGD ($m=1$)"),
]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharex=True, sharey=True)
for ax, (batch_size, title) in zip(axes, method_specs):
    method_path = stochastic_path(batch_size)
    ax.contour(A, B, Z, levels=15, cmap="Blues", linewidths=.8)
    ax.plot(method_path[:, 0], method_path[:, 1], color=orange, lw=1.8, alpha=.9)
    ax.scatter(method_path[::7, 0], method_path[::7, 1], color=orange,
               s=18, zorder=3)
    ax.scatter(method_path[0, 0], method_path[0, 1], color="#263238",
               s=55, zorder=4, label="início")
    ax.scatter(0, beta_star, color=green, marker="*", s=130,
               zorder=4, label="mínimo OLS")
    ax.set_title(title)
    ax.set_xlabel(r"intercepto $\beta_0$")

axes[0].set_ylabel(r"inclinação $\beta_1$")
axes[-1].legend(loc="lower right", fontsize=9)
fig.suptitle("O tamanho do batch controla o ruído do caminho", y=1.01)
save(fig, "caminhos-batch-sgd.png")

# How one shuffled epoch is consumed by batch, mini-batch, and SGD
sample_order = np.array([4, 1, 9, 2, 11, 6, 3, 12, 7, 5, 10, 8])
positions = np.arange(len(sample_order))*1.2
rows = [2, 1, 0]
fig, ax = plt.subplots(figsize=(13, 4.7))

row_specs = [
    ("Batch completo", rows[0], [blue]*12, [11], "1 atualização"),
    ("Mini-batch ($m=4$)", rows[1], [blue]*4+[green]*4+[orange]*4, [3, 7, 11], "3 atualizações"),
    ("SGD ($m=1$)", rows[2], [orange]*12, list(range(12)), "12 atualizações"),
]

for label, y_row, colors, update_after, count_label in row_specs:
    ax.scatter(positions, np.full(12, y_row), marker="s", s=520,
               color=colors, edgecolor="white", linewidth=1.2, zorder=3)
    for x_pos, sample_id in zip(positions, sample_order):
        ax.text(x_pos, y_row, str(sample_id), ha="center", va="center",
                color="white", fontsize=10, fontweight="bold", zorder=4)
    update_x = positions[np.asarray(update_after)] + .52
    ax.scatter(update_x, np.full(len(update_x), y_row), marker="|", s=480,
               color="#263238", linewidth=2.2, zorder=5)
    ax.text(-1.05, y_row, label, ha="right", va="center",
            fontsize=13, color="#17324d", fontweight="bold")
    ax.text(positions[-1]+1.15, y_row, count_label, ha="left", va="center",
            fontsize=11, color="#455a64")

ax.text(positions.mean(), 2.62,
        "Mesma ordem embaralhada em uma época",
        ha="center", va="center", fontsize=17, color="#17324d")
ax.text(positions.mean(), -.63,
        "A barra preta marca uma atualização dos parâmetros",
        ha="center", va="center", fontsize=11, color="#455a64")
ax.set(xlim=(-2.2, positions[-1]+3), ylim=(-.85, 2.85))
ax.axis("off")
save(fig,"ordem-batch-sgd.png")

# scaling effect: contour geometry and gradient paths
def loss_surface(xx, intercepts, slopes):
    intercept_grid, slope_grid = np.meshgrid(intercepts, slopes)
    values = (
        np.mean(yz**2)
        - 2*intercept_grid*np.mean(yz)
        - 2*slope_grid*np.mean(xx*yz)
        + intercept_grid**2
        + 2*intercept_grid*slope_grid*np.mean(xx)
        + slope_grid**2*np.mean(xx**2)
    )
    return intercept_grid, slope_grid, values

def gradient_path(xx, start, learning_rate, steps):
    theta = np.array(start, dtype=float)
    history = []
    for iteration in range(steps):
        prediction = theta[0]+theta[1]*xx
        difference = prediction-yz
        history.append((iteration, *theta, np.mean(difference**2)))
        gradient = np.array([2*np.mean(difference), 2*np.mean(xx*difference)])
        theta -= learning_rate*gradient
    return np.array(history)

raw_intercepts = np.linspace(-1.15, 1.15, 220)
raw_slopes = np.linspace(-.04, .055, 220)
std_intercepts = np.linspace(-1.15, 1.15, 220)
std_slopes = np.linspace(-.65, 1.15, 220)
Ar, Br, Zr = loss_surface(x, raw_intercepts, raw_slopes)
As, Bs, Zs = loss_surface(xz, std_intercepts, std_slopes)

theta_raw = np.linalg.solve(
    np.column_stack([np.ones(len(x)), x]).T @ np.column_stack([np.ones(len(x)), x]),
    np.column_stack([np.ones(len(x)), x]).T @ yz,
)
theta_std = np.linalg.solve(
    np.column_stack([np.ones(len(xz)), xz]).T @ np.column_stack([np.ones(len(xz)), xz]),
    np.column_stack([np.ones(len(xz)), xz]).T @ yz,
)
path_raw = gradient_path(x, [1, -.03], learning_rate=.0005, steps=50000)
path_std = gradient_path(xz, [1, -.5], learning_rate=.2, steps=20)
raw_marks = np.unique(np.r_[np.arange(10), [20, 50, 100, 300, 1000, 3000, 10000, 30000, 49999]])

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
raw_levels = Zr.min() + np.geomspace(.002, 8, 16)
std_levels = Zs.min() + np.geomspace(.002, 4, 16)
axes[0].contour(Ar, Br, Zr, levels=raw_levels, cmap="Blues", linewidths=1.25)
axes[0].plot(path_raw[:, 1], path_raw[:, 2], color=orange, lw=2.2)
axes[0].plot(path_raw[raw_marks, 1], path_raw[raw_marks, 2], "o", color=orange, ms=4)
axes[0].scatter(*theta_raw, s=110, color=green, edgecolor="white", linewidth=1.2, zorder=5, label="mínimo OLS")
axes[0].scatter(path_raw[0, 1], path_raw[0, 2], s=70, color="#263238", zorder=5, label="início")
axes[0].set(
    xlabel=r"intercepto $\beta_0$",
    ylabel=r"inclinação $\beta_1$",
    title="Sem padronização: vale estreito",
    xlim=(raw_intercepts.min(), raw_intercepts.max()),
    ylim=(raw_slopes.min(), raw_slopes.max()),
)
axes[0].legend(loc="upper left", fontsize=10)

zoom = inset_axes(
    axes[0],
    width="40%",
    height="40%",
    loc="upper right",
    borderpad=.8,
)
zoom.contour(Ar, Br, Zr, levels=raw_levels, cmap="Blues", linewidths=.8)
zoom.plot(path_raw[:12, 1], path_raw[:12, 2], "o-", color=orange, lw=1.8, ms=3.5)
zoom.scatter(path_raw[0, 1], path_raw[0, 2], s=38, color="#263238", zorder=5)
zoom.set(xlim=(.992, 1.002), ylim=(-.032, -.011))
zoom.set_title("Primeiras 12 iterações", fontsize=10, pad=3)
zoom.tick_params(labelsize=8)
zoom.grid(alpha=.35)
mark_inset(axes[0], zoom, loc1=1, loc2=4, fc="none", ec="#546e7a", lw=1)

axes[1].contour(As, Bs, Zs, levels=std_levels, cmap="Blues", linewidths=1.25)
axes[1].plot(path_std[:, 1], path_std[:, 2], "o-", color=orange, lw=2.2, ms=4)
axes[1].scatter(*theta_std, s=110, color=green, edgecolor="white", linewidth=1.2, zorder=5, label="mínimo OLS")
axes[1].scatter(path_std[0, 1], path_std[0, 2], s=70, color="#263238", zorder=5, label="início")
axes[1].set(
    xlabel=r"intercepto $\beta_0$",
    ylabel=r"inclinação $\beta_1$",
    title="Com padronização: caminho direto",
    xlim=(std_intercepts.min(), std_intercepts.max()),
    ylim=(std_slopes.min(), std_slopes.max()),
)
axes[1].legend(loc="upper left", fontsize=10)
save(fig,"padronizacao.png")

# Adam on the same unstandardized regression surface
def adam_path(xx, start, learning_rate=.02, steps=300, gamma1=.9, gamma2=.999,
              epsilon=1e-8):
    theta = np.array(start, dtype=float)
    first_moment = np.zeros_like(theta)
    second_moment = np.zeros_like(theta)
    history = []

    for step in range(1, steps+1):
        prediction = theta[0]+theta[1]*xx
        difference = prediction-yz
        gradient = np.array([
            2*np.mean(difference),
            2*np.mean(xx*difference),
        ])
        history.append((step-1, *theta, np.mean(difference**2)))
        first_moment = gamma1*first_moment+(1-gamma1)*gradient
        second_moment = gamma2*second_moment+(1-gamma2)*gradient**2
        corrected_first = first_moment/(1-gamma1**step)
        corrected_second = second_moment/(1-gamma2**step)
        theta -= learning_rate*corrected_first/(np.sqrt(corrected_second)+epsilon)

    return np.asarray(history)

path_adam_raw = adam_path(x, [1, -.03])
path_adam_std = adam_path(xz, [1, -.5])
gd_compare_marks = np.unique(np.r_[
    np.arange(12),
    np.geomspace(12, len(path_raw)-1, 95).astype(int),
])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
raw_comparison_specs = [
    (path_raw[gd_compare_marks], "GD, idade original"),
    (path_adam_raw, "Adam, idade original"),
]
for ax, (method_path, title) in zip(axes[:2], raw_comparison_specs):
    ax.contour(Ar, Br, Zr, levels=raw_levels, cmap="Blues", linewidths=1.15)
    ax.plot(method_path[:, 1], method_path[:, 2], color=orange, lw=2)
    marker_stride = max(1, len(method_path)//18)
    ax.scatter(method_path[::marker_stride, 1], method_path[::marker_stride, 2],
               color=orange, s=20, zorder=3)
    ax.scatter(method_path[-1, 1], method_path[-1, 2], color=orange,
               edgecolor="white", linewidth=1, s=65, zorder=4)
    ax.scatter(method_path[0, 1], method_path[0, 2], s=65,
               color="#263238", zorder=5, label="início")
    ax.scatter(*theta_raw, s=120, color=green, marker="*",
               edgecolor="white", linewidth=1, zorder=5, label="mínimo OLS")
    ax.set(
        xlabel=r"intercepto $\beta_0$",
        ylabel=r"inclinação $\beta_1$",
        title=title,
        xlim=(raw_intercepts.min(), raw_intercepts.max()),
        ylim=(raw_slopes.min(), raw_slopes.max()),
    )

axes[0].text(.03, .04, "50.000 atualizações", transform=axes[0].transAxes,
             fontsize=10, color="#455a64")
axes[1].text(.03, .04, "300 atualizações", transform=axes[1].transAxes,
             fontsize=10, color="#455a64")

axes[2].contour(As, Bs, Zs, levels=std_levels, cmap="Blues", linewidths=1.15)
axes[2].plot(path_adam_std[:, 1], path_adam_std[:, 2], color=orange, lw=2)
axes[2].scatter(path_adam_std[::15, 1], path_adam_std[::15, 2],
                color=orange, s=20, zorder=3)
axes[2].scatter(path_adam_std[-1, 1], path_adam_std[-1, 2], color=orange,
                edgecolor="white", linewidth=1, s=65, zorder=4)
axes[2].scatter(path_adam_std[0, 1], path_adam_std[0, 2], s=65,
                color="#263238", zorder=5, label="início")
axes[2].scatter(*theta_std, s=120, color=green, marker="*",
                edgecolor="white", linewidth=1, zorder=5, label="mínimo OLS")
axes[2].set(
    xlabel=r"intercepto $\beta_0$",
    ylabel=r"inclinação $\beta_1$",
    title="Adam, idade padronizada",
    xlim=(std_intercepts.min(), std_intercepts.max()),
    ylim=(std_slopes.min(), std_slopes.max()),
)
axes[2].text(.03, .04, "300 atualizações", transform=axes[2].transAxes,
             fontsize=10, color="#455a64")
axes[2].legend(loc="upper left", fontsize=9)
fig.suptitle("Escala da entrada e escolha do otimizador", y=1.01)
save(fig, "gd-adam.png")

print({"beta_ols_padronizado":beta_star})
