import torch
import torch.nn as nn
import pandas as pd

from auto_LiRPA import BoundedTensor, BoundedModule, PerturbationLpNorm
import numpy as np


class ResNetTail(nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.tail = nn.Sequential(
            full_model.layer4,
            full_model.avgpool,
            nn.Flatten(),
            full_model.fc
        )

    def forward(self, x):
        return self.tail(x)
    


class ResNetTailLarge(nn.Module):
    def __init__(self, full_model):
        super().__init__()
        # On inclut layer3 et layer4 + avgpool + fc
        self.tail = nn.Sequential(
            full_model.layer3,
            full_model.layer4,
            full_model.avgpool,
            nn.Flatten(),
            full_model.fc
        )

    def forward(self, x):
        return self.tail(x)
    



class ResNetTailVeryLarge(nn.Module):
    def __init__(self, full_model):
        super().__init__()
        # Inclut layer2, layer3, layer4 + avgpool + fc
        self.tail = nn.Sequential(
            full_model.layer2,
            full_model.layer3,
            full_model.layer4,
            full_model.avgpool,
            nn.Flatten(),
            full_model.fc
        )

    def forward(self, x):
        return self.tail(x)

    


def get_mid_activation(x : torch.Tensor, full_model : torch.nn.Module, CHOIX_LAYER_TAIL: int):
    """Extract intermediate activation from full_model at the appropriate layer.
    
    Args:
        x: input tensor
        full_model: the complete neural network (not a tail fragment)
        CHOIX_LAYER_TAIL: which layer to split at (0=no split, 2=after layer1, 3=after layer2, 4=after layer3)
    """
    assert CHOIX_LAYER_TAIL in [0, 2, 3, 4], "Invalid choice for tail layer. Must be 0, 2, 3 or 4."
    if CHOIX_LAYER_TAIL == 0:
        print("Using full model as tail, no mid activation.")
        return x  # On retourne directement l'entrée pour le cas du full model
    
    mid_activation = None

    def _get_mid_activation(module, input, output):
        nonlocal mid_activation  
        # nonlocal signifie que la fonction va modifier la variable mid_activation au niveau du parent de la fonction _get_mid_activation,
        # à savoir get_mid_activation
        mid_activation = output.detach()

    if CHOIX_LAYER_TAIL == 2:
        hook_handle = full_model.layer1.register_forward_hook(_get_mid_activation)
    elif CHOIX_LAYER_TAIL == 3:
        hook_handle = full_model.layer2.register_forward_hook(_get_mid_activation)
    else:  # CHOIX_LAYER_TAIL == 4
        hook_handle = full_model.layer3.register_forward_hook(_get_mid_activation)

    # Forward pass to capture mid activation
    with torch.no_grad():
        _ = full_model(x)

    hook_handle.remove()

    if mid_activation is None:
        raise RuntimeError(f"Failed to capture mid activation at layer choice {CHOIX_LAYER_TAIL}")

    return mid_activation



def create_tail_model(full_model : torch.nn.Module, CHOIX_LAYER_TAIL: int, DEVICE : str):
    assert CHOIX_LAYER_TAIL in [0, 2,3, 4], "Invalid choice for tail layer. Must be 2 or 3."
    if CHOIX_LAYER_TAIL == 0:
        print("Using full model as tail (layer1 + layer2 + layer3 + layer4 + avgpool + fc)")
        tail_model = full_model.to(DEVICE)
        tail_model.eval()
    elif CHOIX_LAYER_TAIL == 2:
        print("Using tail from layer2 (layer2 + layer3 + layer4 + avgpool + fc)")
        tail_model = ResNetTailVeryLarge(full_model).to(DEVICE)
        tail_model.eval()
    elif CHOIX_LAYER_TAIL == 3:
        print("Using large tail (layer3 + layer4 + avgpool + fc)")
        tail_model = ResNetTailLarge(full_model).to(DEVICE)
        tail_model.eval()
    else :
        print("Using standard tail (layer4 + avgpool + fc)")
        tail_model = ResNetTail(full_model).to(DEVICE)
        tail_model.eval()
    return tail_model



def compute_bounds_tail_model(full_model : torch.nn.Module, x : torch.Tensor, 
                   EPSILON : float, DEVICE : str, 
                   NORM, METHOD : str, CHOIX_LAYER_TAIL: int ):
    assert METHOD in ["alpha-CROWN", "CROWN-IBP"], "Invalid method. Must be 'alpha-CROWN' or 'CROWN-IBP'."
    mid_activation = get_mid_activation(x, full_model, CHOIX_LAYER_TAIL)
    mid_activation = mid_activation.to(DEVICE)

    tail_model = create_tail_model(full_model, CHOIX_LAYER_TAIL, DEVICE)

    ptb_mid = PerturbationLpNorm(norm=NORM, eps=EPSILON)
    bounded_mid = BoundedTensor(mid_activation, ptb_mid)
    bounded_tail = BoundedModule(
        tail_model,
        torch.zeros_like(mid_activation).to(DEVICE),
        bound_opts={"conv_mode": "patches"},
    ).to(DEVICE)

    with torch.no_grad():
        #print("Computing bounds from mid-layer...")

        lb_mid, ub_mid = bounded_tail.compute_bounds(
            x=(bounded_mid,),
            method=METHOD
        )

    # print("Lower bounds (tail):", lb_mid)
    # print("Upper bounds (tail):", ub_mid)

    intermediate_bounds_tail = bounded_tail.save_intermediate()
    # print("Gap min : ", (lb_mid - ub_mid).min().item()
    #     )
    # print("Gap max : ", (lb_mid - ub_mid).max().item())
    return intermediate_bounds_tail, bounded_tail

def nb_stable_actives(intermediate_bounds_tail, LAYER):
    print("intermediate_bounds_tail keys dans stable actives:", intermediate_bounds_tail.keys())
    lb = intermediate_bounds_tail[LAYER][0]
    ub = intermediate_bounds_tail[LAYER][1]
    stable_active = ((lb > 0) & (ub > 0)).sum().item()
    return stable_active, stable_active / lb.numel()

def nb_stable_inactives(intermediate_bounds_tail, LAYER):
    lb = intermediate_bounds_tail[LAYER][0]
    ub = intermediate_bounds_tail[LAYER][1]
    stable_inactive = ((lb < 0) & (ub < 0)).sum().item()
    return stable_inactive, stable_inactive / lb.numel()


def detect_ood(full_model : torch.nn.Module, x : torch.Tensor, 
                   EPSILON : float, DEVICE : str, 
                   NORM, METHOD : str, CHOIX_LAYER_TAIL: int, LAYER : str, criterion_stable : str, threshold : float):
    intermediate_bounds_tail, _ = compute_bounds_tail_model(full_model, x, EPSILON, DEVICE, NORM, METHOD, CHOIX_LAYER_TAIL)
    if criterion_stable == "actives":
        nb_stable, ratio_stable = nb_stable_actives(intermediate_bounds_tail, LAYER)
    else:
        nb_stable, ratio_stable = nb_stable_inactives(intermediate_bounds_tail, LAYER)
    #print(f"Number of stable {criterion_stable} neurons at layer {LAYER}: {nb_stable} ({ratio_stable:.2%})")
    is_ood = ratio_stable < threshold
    #print(f"Detection result: {'OOD' if is_ood else 'ID'} (threshold: {threshold:.2%})")
    return is_ood, ratio_stable


def nb_stable_actives_batch(intermediate_bounds_tail: dict, LAYER: str) -> torch.Tensor:
    """Retourne un tenseur de ratios de neurones stables actifs, shape (batch_size,)."""
    lb = intermediate_bounds_tail[LAYER][0]   # (B, C, H, W) ou (B, C)
    ub = intermediate_bounds_tail[LAYER][1]
    n_per_sample = lb[0].numel()
    ratios = ((lb > 0) & (ub > 0)).float().flatten(1).sum(dim=1) / n_per_sample
    return ratios


def nb_stable_inactives_batch(intermediate_bounds_tail: dict, LAYER: str) -> torch.Tensor:
    """Retourne un tenseur de ratios de neurones stables inactifs, shape (batch_size,)."""
    lb = intermediate_bounds_tail[LAYER][0]
    ub = intermediate_bounds_tail[LAYER][1]
    n_per_sample = lb[0].numel()
    ratios = ((lb < 0) & (ub < 0)).float().flatten(1).sum(dim=1) / n_per_sample
    return ratios


def bound_slack_batch(intermediate_bounds_tail: dict, LAYER: str, tail_model: nn.Module, mid: torch.Tensor, sense : str = "upper") -> torch.Tensor:
    """Calcule la marge moyenne ub - h(x) à la couche LAYER, pour chaque sample du batch.

    Pour chaque neurone i : slack_i = ub_i(x) - h_i(x)
    où h_i(x) est la valeur réelle pré-ReLU (= sortie du nœud BoundAdd dans le bloc résiduel).

    Retourne un tenseur de shape (batch_size,) avec la marge moyenne par sample.
    Les valeurs plus élevées indiquent que la borne sup est lâche par rapport à
    l'activation réelle — potentiellement discriminant entre ID et OOD.
    """
    if sense == "upper":
        bound = intermediate_bounds_tail[LAYER][1]  # (B, ...)
    else:
        bound = intermediate_bounds_tail[LAYER][0]  # (B, ...)

    # LAYER est le nœud pré-ReLU (BoundAdd dans les blocs résiduels).
    # On capture sa valeur concrète en hookant le dernier ReLU de tail_model :
    # input[0] de ce ReLU = sortie de l'addition résiduelle = h(x) au nœud LAYER.
    pre_relu_value = {}

    def _hook(module, input, output):
        pre_relu_value['value'] = input[0].detach()

    last_relu = None
    for module in tail_model.modules():
        if isinstance(module, nn.ReLU):
            last_relu = module

    if last_relu is None:
        raise ValueError("Aucun ReLU trouvé dans tail_model.")

    handle = last_relu.register_forward_hook(_hook)
    with torch.no_grad():
        tail_model(mid)
    handle.remove()

    actual = pre_relu_value['value']  # (B, ...)
    if sense == "upper":
        slack = bound - actual  # (B, ...)
    else:
        slack = actual - bound  # (B, ...)
    return slack.flatten(1).mean(dim=1)  # (B,)



def create_statistics_ood_dataset(
    full_model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    EPSILON: float,
    DEVICE: str,
    NORM,
    METHOD: str,
    CHOIX_LAYER_TAIL: int,
    N_LAYERS: int = 5,
    LAYERS: list = None,
    name: str = "",
) -> pd.DataFrame:
    """
    Calcule des statistiques OOD sur les N_LAYERS dernières couches du réseau,
    logits inclus.

    Paramètres
    ----------
    N_LAYERS : int
        Nombre total de points de mesure : (N_LAYERS - 1) couches pré-ReLU + les logits.
        Par défaut 5 → 4 pré-ReLU + logits.
    LAYERS : list[str], optional
        Si fourni, liste des clés de nœuds pré-ReLU à analyser (surpasse N_LAYERS).
        L'ordre doit aller du plus ancien au plus récent dans le réseau.

    Structure du CSV produit
    ------------------------
    Une ligne par sample. Colonnes par couche pré-ReLU avec préfixe L0…L{k} :
      L{i}_ratio_actives_stable, L{i}_ratio_inactives_stable,
      L{i}_gap_{min,mean,max}, L{i}_{min,max,mean}_{lb,ub},
      L{i}_ub_slack_mean, L{i}_lb_slack_mean
    Plus colonnes logits (sans ratio stables) :
      logits_gap_{min,mean,max}, logits_{min,max,mean}_{lb,ub}
    """
    full_model.eval()
    tail_model = create_tail_model(full_model, CHOIX_LAYER_TAIL, DEVICE)

    # Initialisation du BoundedModule avec un dummy sample de taille 1
    dummy_x = next(iter(dataloader))
    dummy_x = dummy_x[0] if isinstance(dummy_x, (list, tuple)) else dummy_x
    dummy_x = dummy_x[0:1].to(DEVICE)
    dummy_mid = get_mid_activation(dummy_x, full_model, CHOIX_LAYER_TAIL).to(DEVICE)

    bounded_tail = BoundedModule(
        tail_model,
        torch.zeros_like(dummy_mid),
        bound_opts={"conv_mode": "patches"},
    ).to(DEVICE)

    # Auto-détection des couches pré-ReLU si non fournies
    if LAYERS is None:
        ptb_dummy = PerturbationLpNorm(norm=NORM, eps=EPSILON)
        bounded_tail.compute_bounds(x=(BoundedTensor(dummy_mid, ptb_dummy),), method=METHOD)
        n_prerelu = N_LAYERS - 1  # N_LAYERS - 1 pré-ReLU + logits = N_LAYERS points
        LAYERS = get_last_N_prerelu_layers(bounded_tail, n_prerelu)
        print(f"Couches pré-ReLU sélectionnées ({len(LAYERS)}) : {LAYERS}")

    n_prerelu = len(LAYERS)
    # L0 = plus ancien des sélectionnés, L{n_prerelu-1} = dernier avant logits
    layer_prefixes = [f"L{i}" for i in range(n_prerelu)]

    # Hooks persistants sur les derniers n_prerelu ReLUs de tail_model.
    # L'hypothèse est que l'ordre de parcours de modules() correspond à l'ordre
    # d'exécution forward, ce qui est vrai pour les ResNets standards.
    relu_list = [m for m in tail_model.modules() if isinstance(m, nn.ReLU)]
    if len(relu_list) < n_prerelu:
        raise ValueError(
            f"tail_model n'a que {len(relu_list)} ReLUs, impossible d'en hooker {n_prerelu}."
        )

    pre_relu_captures: dict = {}  # local_idx -> tensor CPU (B, ...)
    hook_handles = []
    for local_idx, relu_mod in enumerate(relu_list[-n_prerelu:]):
        def make_hook(idx: int):
            def hook(mod, inp, out):
                pre_relu_captures[idx] = inp[0].detach().cpu()
            return hook
        hook_handles.append(relu_mod.register_forward_hook(make_hook(local_idx)))

    # Accumulateurs per-neurone (pour stats globales sur le dataset)
    neuron_acc = {
        layer: {"stable_active": None, "stable_inactive": None, "actual_active": None}
        for layer in LAYERS
    }

    tab_stats = pd.DataFrame()
    n_total = 0
    n_max = 1000

    for batch_idx, batch in enumerate(dataloader):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, labels = batch
        else:
            inputs, labels = batch, None

        inputs = inputs.to(DEVICE)
        B = inputs.shape[0]

        mid = get_mid_activation(inputs, full_model, CHOIX_LAYER_TAIL).to(DEVICE)
        bounded_mid = BoundedTensor(mid, PerturbationLpNorm(norm=NORM, eps=EPSILON))

        with torch.no_grad():
            lb_out, ub_out = bounded_tail.compute_bounds(x=(bounded_mid,), method=METHOD)

        inter_bounds = bounded_tail.save_intermediate()

        # Forward pass explicite pour peupler les hooks pré-ReLU (h(x) concret)
        with torch.no_grad():
            logits_actual = tail_model(mid)
        predicted_labels = logits_actual.argmax(dim=1).cpu().numpy()

        row_dict: dict = {
            "batch_idx":       np.full(B, batch_idx),
            "epsilon":         np.full(B, EPSILON),
            "method":          [METHOD] * B,
            "layer_tail_nb":   np.full(B, CHOIX_LAYER_TAIL),
            "sample_idx":      np.arange(n_total, n_total + B),
            "predicted_label": predicted_labels,
        }
        if labels is not None:
            row_dict["true_label"] = labels.cpu().numpy()

        # ── Stats par couche pré-ReLU ──────────────────────────────────────────
        for local_idx, (layer_name, prefix) in enumerate(zip(LAYERS, layer_prefixes)):
            stats = compute_layer_stats_batch(inter_bounds, layer_name)
            for metric, values in stats.items():
                row_dict[f"{prefix}_{metric}"] = values.numpy()

            # Slack : ub - h(x) et h(x) - lb  (mesure de la laxité des bornes)
            if local_idx in pre_relu_captures:
                hx   = pre_relu_captures[local_idx].flatten(start_dim=1)          # (B, N)
                lb_f = inter_bounds[layer_name][0].cpu().flatten(start_dim=1)      # (B, N)
                ub_f = inter_bounds[layer_name][1].cpu().flatten(start_dim=1)      # (B, N)
                row_dict[f"{prefix}_ub_slack_mean"] = (ub_f - hx).mean(dim=1).numpy()
                row_dict[f"{prefix}_lb_slack_mean"] = (hx - lb_f).mean(dim=1).numpy()
            else:
                row_dict[f"{prefix}_ub_slack_mean"] = np.full(B, np.nan)
                row_dict[f"{prefix}_lb_slack_mean"] = np.full(B, np.nan)

            # Accumulation per-neurone (certifié par alpha-beta-CROWN)
            lb_f = inter_bounds[layer_name][0].cpu().flatten(start_dim=1)
            ub_f = inter_bounds[layer_name][1].cpu().flatten(start_dim=1)
            sa = ((lb_f > 0) & (ub_f > 0))
            si = ((lb_f < 0) & (ub_f < 0))
            if neuron_acc[layer_name]["stable_active"] is None:
                neuron_acc[layer_name]["stable_active"]   = sa.sum(dim=0)
                neuron_acc[layer_name]["stable_inactive"] = si.sum(dim=0)
            else:
                neuron_acc[layer_name]["stable_active"]   += sa.sum(dim=0)
                neuron_acc[layer_name]["stable_inactive"] += si.sum(dim=0)

            if local_idx in pre_relu_captures:
                hx_acc = pre_relu_captures[local_idx].flatten(start_dim=1)
                aa = (hx_acc > 0)
                if neuron_acc[layer_name]["actual_active"] is None:
                    neuron_acc[layer_name]["actual_active"] = aa.sum(dim=0)
                else:
                    neuron_acc[layer_name]["actual_active"] += aa.sum(dim=0)

        # ── Stats logits ───────────────────────────────────────────────────────
        logits_stats = compute_logits_stats_batch(lb_out, ub_out)
        for metric, values in logits_stats.items():
            row_dict[f"logits_{metric}"] = values.numpy()

        tab_stats = pd.concat([tab_stats, pd.DataFrame(row_dict)], ignore_index=True)

        n_total += B
        if n_total >= n_max:
            print(f"Limite de {n_max} échantillons atteinte, arrêt.")
            break

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx} — {n_total} samples traités.")

    for h in hook_handles:
        h.remove()

    # ── Sauvegarde CSV ─────────────────────────────────────────────────────────
    layers_tag = f"last{n_prerelu}relu_logits"
    csv_path = (
        f"/share/homes/boyerma/robustesse_ood/results/"
        f"statistics_ood_{layers_tag}_{name}_{n_total}_datas.csv"
    )
    tab_stats.to_csv(csv_path, index=False)
    print(f"CSV sauvegardé : {csv_path}")

    # ── Sauvegarde stats per-neurone ───────────────────────────────────────────
    for layer_name in LAYERS:
        layer_tag = layer_name.replace('/', '_')
        prefix_path = (
            f"/share/homes/boyerma/robustesse_ood/results/"
            f"neuron_stats_layer{layer_tag}_{name}_{n_total}"
        )
        acc = neuron_acc[layer_name]
        if acc["stable_active"] is not None:
            np.save(f"{prefix_path}_stable_active.npy",   (acc["stable_active"]   / n_total).numpy())
            np.save(f"{prefix_path}_stable_inactive.npy", (acc["stable_inactive"] / n_total).numpy())
        if acc["actual_active"] is not None:
            np.save(f"{prefix_path}_actual_active.npy",   (acc["actual_active"]   / n_total).numpy())

    print(f"Neuron stats sauvegardées pour {len(LAYERS)} couches pré-ReLU.")
    return tab_stats



def detect_ood_dataset(
    full_model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    EPSILON: float,
    DEVICE: str,
    NORM,
    METHOD: str,
    CHOIX_LAYER_TAIL: int,
    criterion_stable: str,
    threshold: float,
    LAYER: str = None,
    verbose: bool = False,
) -> dict:
    assert criterion_stable in ["actives", "inactives"], \
        "criterion_stable must be 'actives' or 'inactives'."

    full_model.eval()
    tail_model = create_tail_model(full_model, CHOIX_LAYER_TAIL, DEVICE)

    # BoundedModule initialisé une fois avec un sample de taille 1
    dummy_x = next(iter(dataloader))
    dummy_x = dummy_x[0] if isinstance(dummy_x, (list, tuple)) else dummy_x
    dummy_x = dummy_x[0:1].to(DEVICE)
    dummy_mid = get_mid_activation(dummy_x, full_model, CHOIX_LAYER_TAIL).to(DEVICE)

    bounded_tail = BoundedModule(
        tail_model,
        torch.zeros_like(dummy_mid),
        bound_opts={"conv_mode": "patches"},
    ).to(DEVICE)

    # Résoudre LAYER une seule fois si non fourni
    if LAYER is None:
        ptb_dummy = PerturbationLpNorm(norm=NORM, eps=EPSILON)
        bounded_tail.compute_bounds(x=(BoundedTensor(dummy_mid, ptb_dummy),), method=METHOD)
        LAYER = get_last_prerelu_layer(bounded_tail)
        print(f"LAYER auto-détecté : {LAYER}")

    is_ood_list, ratio_list, label_list = [], [], []
    n_total = 0
    n_max = 100
    has_labels = False

    for batch_idx, batch in enumerate(dataloader):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, labels = batch
            has_labels = True
        else:
            inputs, labels = batch, None

        inputs = inputs.to(DEVICE)  # (B, C, H, W)

        # --- Un seul appel LiRPA pour tout le batch ---
        mid = get_mid_activation(inputs, full_model, CHOIX_LAYER_TAIL).to(DEVICE)
        bounded_mid = BoundedTensor(mid, PerturbationLpNorm(norm=NORM, eps=EPSILON))

        with torch.no_grad():
            bounded_tail.compute_bounds(x=(bounded_mid,), method=METHOD)

        inter_bounds = bounded_tail.save_intermediate()

        if criterion_stable == "actives":
            ratios = nb_stable_actives_batch(inter_bounds, LAYER)   # (B,)
        else:
            ratios = nb_stable_inactives_batch(inter_bounds, LAYER) # (B,)

        batch_is_ood = (ratios < threshold).tolist()
        batch_ratios = ratios.tolist()

        is_ood_list.extend(batch_is_ood)
        ratio_list.extend(batch_ratios)
        if has_labels:
            label_list.extend(labels.tolist())

        n_total += inputs.shape[0]
        if n_total >= n_max:
            print(f"Limite de {n_max} échantillons atteinte, arrêt du calcul LiRPA pour les batches suivants.")
            break

        if verbose:
            for i, (r, ood) in enumerate(zip(batch_ratios, batch_is_ood)):
                print(f"  [{n_total - inputs.shape[0] + i}] ratio={r:.4f} -> {'OOD' if ood else 'ID'}")

        print(f"Batch {batch_idx + 1}/{len(dataloader)} — "
              f"{n_total} samples, OOD rate: {sum(is_ood_list)/n_total:.2%}")

    ood_rate = sum(is_ood_list) / n_total if n_total > 0 else 0.0
    print(f"\n=== Résumé détection OOD ===")
    print(f"Samples total : {n_total}")
    print(f"Taux OOD      : {ood_rate:.2%}  (seuil={threshold:.2%})")

    return {
        "is_ood":    is_ood_list,
        "ratios":    ratio_list,
        "labels":    label_list if has_labels else None,
        "ood_rate":  ood_rate,
        "n_samples": n_total,
    }


def get_last_prerelu_layer(bounded_tail: BoundedModule) -> str:
    """
    Retourne la clé (nom) du nœud juste avant la dernière ReLU
    dans le graphe de bounded_tail — à utiliser comme LAYER dans
    nb_stable_actives / nb_stable_inactives.
    """
    last_relu_node = None
    for node in bounded_tail.nodes():
        if "Relu" in type(node).__name__:
            last_relu_node = node

    if last_relu_node is None:
        raise ValueError("Aucune ReLU trouvée dans le BoundedModule.")

    pre_relu_node = last_relu_node.inputs[0]
    return pre_relu_node.name


def get_last_N_prerelu_layers(bounded_tail: BoundedModule, N: int = 4) -> list:
    """
    Retourne les noms des N nœuds juste avant les N dernières ReLUs
    dans le graphe de bounded_tail, ordonnés du plus ancien au plus récent.
    """
    relu_nodes = [node for node in bounded_tail.nodes() if "Relu" in type(node).__name__]
    if len(relu_nodes) < N:
        print(f"Warning: seulement {len(relu_nodes)} ReLUs trouvées, N réduit à {len(relu_nodes)}.")
        N = len(relu_nodes)
    selected = relu_nodes[-N:]
    return [node.inputs[0].name for node in selected]


def compute_layer_stats_batch(inter_bounds: dict, layer_name: str) -> dict:
    """
    Calcule les statistiques (ratios stables, gap, lb, ub) pour une couche pré-ReLU.
    Retourne un dict de tenseurs CPU de shape (B,), correctement agrégés par sample.
    """
    lb = inter_bounds[layer_name][0].cpu().flatten(start_dim=1)  # (B, N)
    ub = inter_bounds[layer_name][1].cpu().flatten(start_dim=1)  # (B, N)
    gap = ub - lb                                                 # (B, N)
    stable_active   = ((lb > 0) & (ub > 0)).float()
    stable_inactive = ((lb < 0) & (ub < 0)).float()
    return {
        "ratio_actives_stable":   stable_active.mean(dim=1),
        "ratio_inactives_stable": stable_inactive.mean(dim=1),
        "gap_min":  gap.min(dim=1)[0],
        "gap_mean": gap.mean(dim=1),
        "gap_max":  gap.max(dim=1)[0],
        "min_lb": lb.min(dim=1)[0],
        "max_lb": lb.max(dim=1)[0],
        "mean_lb": lb.mean(dim=1),
        "min_ub": ub.min(dim=1)[0],
        "max_ub": ub.max(dim=1)[0],
        "mean_ub": ub.mean(dim=1),
    }


def compute_logits_stats_batch(lb_logits: torch.Tensor, ub_logits: torch.Tensor) -> dict:
    """
    Calcule les statistiques sur les logits (sortie finale, pas de ReLU).
    Pas de ratio stables/inactifs : seulement gap et lb/ub.
    Retourne un dict de tenseurs CPU de shape (B,).
    """
    lb  = lb_logits.cpu()   # (B, C)
    ub  = ub_logits.cpu()
    gap = ub - lb
    return {
        "gap_min":  gap.min(dim=1)[0],
        "gap_mean": gap.mean(dim=1),
        "gap_max":  gap.max(dim=1)[0],
        "min_lb": lb.min(dim=1)[0],
        "max_lb": lb.max(dim=1)[0],
        "mean_lb": lb.mean(dim=1),
        "min_ub": ub.min(dim=1)[0],
        "max_ub": ub.max(dim=1)[0],
        "mean_ub": ub.mean(dim=1),
    }