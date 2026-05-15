import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ============================================================
# Permutazioni come liste:
# p[i] = immagine di i
# ============================================================


def compose(p, q):
    """p o q"""
    n = len(p)
    return [p[q[i]] for i in range(n)]


def inverse(p):
    n = len(p)
    inv = [0] * n
    for i, x in enumerate(p):
        inv[x] = i
    return inv


def cycle_count(p):
    n = len(p)
    vis = [False] * n
    cnt = 0

    for i in range(n):
        if not vis[i]:
            cnt += 1
            x = i
            while not vis[x]:
                vis[x] = True
                x = p[x]

    return cnt


def is_single_cycle(p):
    return cycle_count(p) == 1


# ============================================================
# Shift S(i)=i+1 mod n
# ============================================================

def make_shift(n):
    return [(i + 1) % n for i in range(n)]


# ============================================================
# Involuzione senza punti fissi
# rappresentata come permutazione
# ============================================================

def random_involution(n):
    arr = list(range(n))
    random.shuffle(arr)

    M = [-1] * n

    for i in range(0, n, 2):
        a = arr[i]
        b = arr[i + 1]

        M[a] = b
        M[b] = a

    return M


# ============================================================
# Mossa locale:
# (a b)(c d) -> (a c)(b d)
# oppure -> (a d)(b c)
# ============================================================

def rewire(M):
    n = len(M)

    a = random.randrange(n)
    b = M[a]

    while True:
        c = random.randrange(n)
        if c != a and c != b:
            break

    d = M[c]

    # evita degenerazioni
    if d == a or d == b:
        return

    if random.random() < 0.5:
        # (a c)(b d)
        M[a] = c
        M[c] = a
        M[b] = d
        M[d] = b
    else:
        # (a d)(b c)
        M[a] = d
        M[d] = a
        M[b] = c
        M[c] = b


# ============================================================
# Costruzione T da P=SM
#
# Se P è un n-ciclo:
# T(k)=P^k(0)
# ============================================================

def build_T_from_cycle(P):
    n = len(P)

    T = [0] * n

    x = 0
    for k in range(n):
        T[k] = x
        x = P[x]

    return T


# ============================================================
# Energia
# ============================================================

def energy(M, S, C):
    n = len(M)

    # P = S M
    P = compose(S, M)

    # termine principale:
    e1 = cycle_count(P) - 1

    # se non è ciclo unico, penalità forte
    if e1 > 0:
        return 1000 * e1, None

    # costruisci T
    T = build_T_from_cycle(P)

    TC = set(T[x] for x in C)

    e2 = len(TC.symmetric_difference(C))

    return e2, T


# ============================================================
# Simulated annealing
# ============================================================

def solve(W, H,
          steps=500000,
          temp0=5.0,
          cooling=0.99995):

    n = 2 * (W + H) - 4
    assert n % 4 == 0, f"n={n} non è multiplo di 4 (W+H deve essere pari)"

    S = make_shift(n)

    C = {0, W - 1, W + H - 2, 2 * W + H - 3}

    M = random_involution(n)

    E, T = energy(M, S, C)

    bestM = M[:]
    bestE = E
    bestT = T
    best_step = 0

    temp = temp0

    for step in range(steps):

        oldM = M[:]

        rewire(M)

        newE, newT = energy(M, S, C)

        delta = newE - E

        accept = False

        if delta <= 0:
            accept = True
        else:
            prob = math.exp(-delta / temp)
            if random.random() < prob:
                accept = True

        if accept:
            E = newE
            T = newT

            if E < bestE:
                bestE = E
                bestM = M[:]
                bestT = T
                best_step = step

                print("step", step, "bestE", bestE)

                if bestE == 0:
                    break

        else:
            M = oldM

        temp *= cooling

    return bestM, bestT, bestE, best_step


# ============================================================
# Verifica finale
# ============================================================

def verify(T, W, H):
    n = 2 * (W + H) - 4
    S = make_shift(n)

    Tin = inverse(T)
    Sin = inverse(S)

    M = compose(Sin, compose(Tin, compose(S, T)))

    MM = compose(M, M)
    ok1 = all(MM[i] == i for i in range(n))
    ok2 = all(M[i] != i for i in range(n))

    C = {0, W - 1, W + H - 2, 2 * W + H - 3}
    TC = {T[x] for x in C}
    ok3 = (TC == C)

    return ok1, ok2, ok3


# ============================================================
# Visualizzazione
# ============================================================

def visualize(T, M, W, H):
    """
    Due griglie affiancate: gli n=2(W+H)-4 elementi disposti sul perimetro
    di un rettangolo W×H, senso orario da TL.

    Griglia sinistra : etichetta = indice i, bordi colorati per coppie M
    Griglia destra   : etichetta = T[i]
    """
    n = 2 * (W + H) - 4

    def get_pos(i):
        if i < W - 1:               # top: sinistra → destra
            return (i, H - 1)
        elif i < W + H - 2:         # right: alto → basso
            return (W - 1, W + H - 2 - i)
        elif i < 2 * W + H - 3:     # bottom: destra → sinistra
            return (2 * W + H - 3 - i, 0)
        else:                        # left: basso → alto
            return (0, i - (2 * W + H - 3))

    # ------------------------------------------------------------------
    # Colore del bordo i: border_i e border_{M[i]} condividono il colore.
    # M è un'involuzione, quindi le coppie {i, M[i]} partizionano {0..n-1}.
    # Assegniamo un indice colore al minore di ogni coppia.
    # ------------------------------------------------------------------
    border_color_idx = [-1] * n
    num_groups = 0
    for i in range(n):
        if border_color_idx[i] == -1:
            border_color_idx[i] = num_groups
            border_color_idx[M[i]] = num_groups
            num_groups += 1

    cmap = plt.colormaps['hsv'].resampled(num_groups)
    border_colors = [cmap(border_color_idx[i]) for i in range(n)]

    # ------------------------------------------------------------------
    # Segmento del bordo i: lato condiviso tra cella i e cella (i+1)%n
    # ------------------------------------------------------------------
    def border_segment(i):
        x1, y1 = get_pos(i)
        x2, y2 = get_pos((i + 1) % n)
        if y1 == y2:
            xm = (x1 + x2) / 2
            return (xm, y1 - 0.5), (xm, y1 + 0.5)
        else:
            ym = (y1 + y2) / 2
            return (x1 - 0.5, ym), (x1 + 0.5, ym)

    # ------------------------------------------------------------------
    # Disegno
    # ------------------------------------------------------------------
    fontsize = max(5, 11 - n // 32)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    # per griglia 1: label[i]=i  → colore bordo i = border_colors[i]
    # per griglia 2: label[i]=T[i] → colore bordo i = border_colors[T[i]]
    configs = [
        (axes[0], list(range(n)), f"Indici (senso orario)  —  n={n}", list(range(n))),
        (axes[1], T,              "Permutazione T(i) per ogni cella i",  T),
    ]

    for ax, labels, title, label_order in configs:
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(-0.5, H - 0.5)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=11, pad=10)
        ax.axis('off')

        for i in range(n):
            x, y = get_pos(i)
            ax.add_patch(patches.Rectangle(
                (x - 0.5, y - 0.5), 1.0, 1.0,
                linewidth=0, facecolor='#AED6F1', zorder=2
            ))
            ax.text(x, y, str(labels[i]),
                    ha='center', va='center', fontsize=fontsize, zorder=3)

        for i in range(n):
            (bx1, by1), (bx2, by2) = border_segment(i)
            ax.plot([bx1, bx2], [by1, by2],
                    color=border_colors[label_order[i]], linewidth=2.5,
                    zorder=4, solid_capstyle='butt')

    plt.tight_layout()
    plt.savefig("puzzle_solution.png", dpi=150, bbox_inches='tight')
    print(f"Figura salvata in puzzle_solution.png  ({num_groups} coppie di bordi)")
    plt.show()


# ============================================================
# Esempio
# ============================================================

if __name__ == "__main__":

    SHOW_PLOT = False

    import time

    # W=H → n = 4*(W-1), multiplo di 4 per ogni W≥2
    sizes = [2, 4, 6, 8, 10, 12, 14, 16]

    for W in sizes:
        H = W
        n = 2 * (W + H) - 4
        threshold = n * n * (n - 1) * (n - 2) * (n - 3)

        t0 = time.time()
        M, T, E, best_step = solve(W, H)
        elapsed = time.time() - t0

        solved = (E == 0)
        print(f"W=H={W:2d}  n={n:3d}  solved={solved}  E={E}  time={elapsed:.2f}s", end="")
        if solved:
            print(f"  |  step={best_step}  estimate={threshold:.2e}  ratio={best_step/threshold:.4f}", end="")
        print()

        if T is not None:
            ok = verify(T, W, H)
            print(f"         VERIFY={ok}")

            if solved and SHOW_PLOT:
                visualize(T, M, W, H)