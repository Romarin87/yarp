#!/usr/bin/env python3
"""
Reaction enumeration script (break then form).

Mimics the flow in tests/tutorial/enumeration_tutorial.py:
    1) Start from one或多个反应物
    2) 可选先断 n 条键（默认 2）
    3) 再做 n 次成键扩展（默认 2）
全程基于哈希去重，并输出带原子映射的 SMILES 以及反应物/产物对应关系。
"""

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import repeat
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import yarp as yp
from rdkit import Chem

# 最高允许键级（超过将被过滤）
MAX_BOND_ORDER = 3

# bond order -> RDKit bond type 映射（限制到三键）
BOND_TYPE = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
}


class RunLogger:
    """同时写文件和 stdout 的简单日志器。"""

    def __init__(self, path: str):
        self.path = path
        self.handle = open(path, "w", encoding="utf-8")

    def log(self, msg: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line)
        self.handle.write(line + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def yarpecule_to_mapped_smiles(y: yp.yarpecule) -> str:
    """
    用最低能的 bond_mat 生成带原子映射的 SMILES，映射号按当前索引（从 1 开始）。
    """
    bmat = y.bond_mats[0]
    fc = yp.return_formals(bmat, y.elements)
    mol = Chem.RWMol()

    for idx, el in enumerate(y.elements):
        atom = Chem.Atom(el.capitalize())
        atom.SetProp("molAtomMapNumber", str(idx + 1))
        atom.SetFormalCharge(int(fc[idx]))
        atom.SetNumRadicalElectrons(int(bmat[idx, idx] % 2))
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
        mol.AddAtom(atom)

    for i in range(len(bmat)):
        for j in range(i):
            order = int(bmat[i, j])
            if order <= 0:
                continue
            bond_type = BOND_TYPE.get(order)
            if bond_type is None:
                raise ValueError(f"Unsupported bond order {order} between atoms {i} and {j}")
            mol.AddBond(i, j, bond_type)

    mol = mol.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass

    return Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True)


def has_high_order_bond(bmat, max_order: int = MAX_BOND_ORDER) -> bool:
    """检测是否存在超过 max_order 的键（仅看非对角元）。"""
    size = len(bmat)
    for i in range(size):
        for j in range(i):
            if bmat[i, j] > max_order:
                return True
    return False


def build_seeds(raw: Iterable[str]) -> List[yp.yarpecule]:
    """
    生成 yarpecule 列表；canon=False 以保留输入顺序用于原子映射。
    """
    seeds = []
    for item in raw:
        if isinstance(item, yp.yarpecule):
            seeds.append(item)
        else:
            try:
                seeds.append(yp.yarpecule(item, canon=False))
            except Exception as e:
                raise ValueError(f"Failed to build yarpecule from input '{item}': {e}")
    return seeds


def _break_bonds_worker(parent: yp.yarpecule, n_break: int) -> List[yp.yarpecule]:
    """单个分子的断键阶段（本地去重），供并行 map 使用。"""
    mids: List[yp.yarpecule] = []
    mid_hashes = set()
    produced = False
    for mid in yp.break_bonds(parent, n=n_break, hashes=mid_hashes, remove_redundant=True):
        produced = True
        mids.append(mid)
    return mids if produced else []


def _form_bonds_worker(mol: yp.yarpecule, inter: bool, intra: bool) -> List[yp.yarpecule]:
    """单个分子的成键阶段（本地去重），供并行 map 使用。"""
    prods: List[yp.yarpecule] = []
    step_hashes = set()
    for prod in yp.form_bonds(
        mol,
        hashes=step_hashes,
        inter=inter,
        intra=intra,
        hash_filter=True,
    ):
        prods.append(prod)
    return prods


def enumerate_reaction_mode(
    seeds: Sequence[yp.yarpecule],
    n_break: int = 2,
    n_form: int = 2,
    allow_break: bool = True,
    inter: bool = True,
    intra: bool = True,
    score_thresh: float = None,
    max_iter: int = None,
    product_hashes: Optional[Set[float]] = None,
    reaction_pairs: Optional[Set[Tuple[float, float]]] = None,
    mapped_cache: Optional[Dict[float, str]] = None,
    score_cache: Optional[Dict[float, float]] = None,
    logger: Optional[RunLogger] = None,
) -> Tuple[List[yp.yarpecule], List[Tuple[float, float, str]], dict]:
    """
    单一 (n_break, n_form) 组合的断键/成键枚举。
    每轮：断 n_break 条键（可选）+ 连续成 n_form 条键，直到无新增或达到 max_iter。
    只记录每一轮的最终产物（不保存中间的断键/单步成键结果）。
    返回：
        products: 本次新增的唯一分子（首次调用时包含种子）
        edges: (parent_hash, child_hash, action) 列表，action 标识完整一轮，形如 cycle{idx}_b{n_break}_f{n_form}
        mapped_cache: hash -> atom-mapped SMILES
        score_cache: hash -> bond_mat_scores[0]
    过滤：跳过包含大于 MAX_BOND_ORDER 的键、score 超阈值或形式电荷不合规的结构。
    product_hashes / reaction_pairs / mapped_cache / score_cache 可传入共用，用于跨模式去重。
    反应去重：用无向键 (min(parent,child), max(parent,child)) 作为反应哈希，正逆视为同一个反应。
    """
    product_hashes = product_hashes if product_hashes is not None else set()
    reaction_pairs = reaction_pairs if reaction_pairs is not None else set()  # 存储无向反应端点，避免重复计入同一反应
    mapped_cache = mapped_cache if mapped_cache is not None else {}
    score_cache = score_cache if score_cache is not None else {}
    products: List[yp.yarpecule] = []
    for y in seeds:
        if y.hash not in product_hashes:
            product_hashes.add(y.hash)
            products.append(y)
        if y.hash not in mapped_cache:
            mapped_cache[y.hash] = yarpecule_to_mapped_smiles(y)
        if y.hash not in score_cache:
            score_cache[y.hash] = y.bond_mat_scores[0]
    edges: List[Tuple[float, float, str]] = []

    parents = list(seeds)
    iter_idx = 0

    if logger:
        logger.log("-" * 12 + f" b{n_break}f{n_form} " + "-" * 12)
        logger.log(f"[b{n_break}f{n_form}] start | seeds={len(seeds)}")
    mode_start = time.time()

    while parents:
        if max_iter is not None and iter_idx >= max_iter:
            break
        iter_idx += 1
        action_label = f"cycle{iter_idx}_b{n_break}_f{n_form}"

        # 1) 断键阶段（仅用于生成中间体，不计入最终产品集合）
        intermediates: List[Tuple[yp.yarpecule, yp.yarpecule]] = []
        if allow_break and n_break > 0:
            mid_hashes = set()  # 用于去重中间体
            for parent in parents:
                produced = False
                for mid in yp.break_bonds(parent, n=n_break, hashes=mid_hashes, remove_redundant=True):
                    produced = True
                    intermediates.append((parent, mid))
                # 若没断出结果，保留原分子参与后续成键
                if not produced:
                    intermediates.append((parent, parent))
        else:
            intermediates = [(p, p) for p in parents]

        if not intermediates:
            break

        # 2) 成键阶段：连续 n_form 轮，每轮从前一轮结果继续成键
        frontier = intermediates  # 每项是 (origin_parent, current_mol)
        for _ in range(max(n_form, 0)):
            next_frontier: List[Tuple[yp.yarpecule, yp.yarpecule]] = []
            step_hashes = set()
            for origin, mol in frontier:
                for prod in yp.form_bonds(
                    mol,
                    hashes=step_hashes,
                    inter=inter,
                    intra=intra,
                    hash_filter=True,
                ):
                    next_frontier.append((origin, prod))
            frontier = next_frontier
            if not frontier:
                break

        # 3) 收集一轮的最终产物，并为下一轮设置 parents
        new_parents: List[yp.yarpecule] = []
        for origin, prod in frontier if n_form > 0 else intermediates:
            bmat = prod.bond_mats[0]
            if has_high_order_bond(bmat):
                continue
            if score_thresh is not None and prod.bond_mat_scores[0] > score_thresh:
                continue
            # formal charge filters
            fc = yp.return_formals(bmat, prod.elements)
            if any(abs(chg) > 1 for chg in fc):
                continue
            if sum(1 for chg in fc if chg != 0) > 2:
                continue
            # 排除反应物与产物相同的情况
            if origin.hash == prod.hash:
                continue

            # 反应去重（正逆同一视为重复）
            rxn_key = tuple(sorted((origin.hash, prod.hash)))
            if rxn_key in reaction_pairs:
                continue
            reaction_pairs.add(rxn_key)

            # 即便产物已存在，也记录新的反应边，但不重复加入节点/parents
            if prod.hash not in product_hashes:
                product_hashes.add(prod.hash)
                mapped_cache[prod.hash] = yarpecule_to_mapped_smiles(prod)
                score_cache[prod.hash] = prod.bond_mat_scores[0]
                products.append(prod)
                new_parents.append(prod)
            else:
                if prod.hash not in mapped_cache:
                    mapped_cache[prod.hash] = yarpecule_to_mapped_smiles(prod)
                if prod.hash not in score_cache:
                    score_cache[prod.hash] = prod.bond_mat_scores[0]

            edges.append((origin.hash, prod.hash, action_label))

        parents = new_parents

    if logger:
        elapsed = time.time() - mode_start
        logger.log(f"[b{n_break}f{n_form}] done | +{len(products)} products | +{len(edges)} edges | elapsed={elapsed:.2f}s")

    return products, edges, mapped_cache, score_cache


def enumerate_reaction_mode_by_seed(
    seeds: Sequence[Union[yp.yarpecule, str]],
    n_break: int = 2,
    n_form: int = 2,
    allow_break: bool = True,
    inter: bool = True,
    intra: bool = True,
    score_thresh: float = None,
    max_iter: int = None,
    product_hashes: Optional[Set[float]] = None,
    reaction_pairs: Optional[Set[Tuple[float, float]]] = None,
    mapped_cache: Optional[Dict[float, str]] = None,
    score_cache: Optional[Dict[float, float]] = None,
    num_workers: int = 1,
    logger: Optional[RunLogger] = None,
) -> Tuple[List[yp.yarpecule], List[Tuple[float, float, str]], dict]:
    """
    map-reduce 模式：并行生成候选，在主进程去重/过滤，保证与串行相同的去重结果。
    num_workers<=1 时退化为原有的 enumerate_reaction_mode。
    """
    if num_workers is None or num_workers < 1:
        num_workers = 1

    # 保持与原接口一致：允许传入 str 或 yarpecule
    seed_objs: List[yp.yarpecule] = []
    for item in seeds:
        if isinstance(item, yp.yarpecule):
            seed_objs.append(item)
        else:
            seed_objs.append(yp.yarpecule(item, canon=False))

    # 单线程/单 seed 时直接走原函数以复用日志输出
    if num_workers == 1 or len(seed_objs) == 1:
        return enumerate_reaction_mode(
            seed_objs,
            n_break=n_break,
            n_form=n_form,
            allow_break=allow_break,
            inter=inter,
            intra=intra,
            score_thresh=score_thresh,
            max_iter=max_iter,
            product_hashes=product_hashes,
            reaction_pairs=reaction_pairs,
            mapped_cache=mapped_cache,
            score_cache=score_cache,
            logger=logger,
        )

    if logger:
        logger.log("-" * 12 + f" b{n_break}f{n_form} " + "-" * 12)
        logger.log(f"[b{n_break}f{n_form}] start | seeds={len(seed_objs)} | workers={num_workers}")
        logger.log(f"[b{n_break}f{n_form}] parallel map-reduce | seeds={len(seed_objs)} | workers={num_workers}")

    product_hashes = product_hashes if product_hashes is not None else set()
    reaction_pairs = reaction_pairs if reaction_pairs is not None else set()
    mapped_cache = mapped_cache if mapped_cache is not None else {}
    score_cache = score_cache if score_cache is not None else {}
    mode_start = time.time()

    products: List[yp.yarpecule] = []
    edges: List[Tuple[float, float, str]] = []

    # 先记录种子以避免在汇总时误判为新产物
    for y in seed_objs:
        if y.hash not in product_hashes:
            product_hashes.add(y.hash)
            products.append(y)
        if y.hash not in mapped_cache:
            mapped_cache[y.hash] = yarpecule_to_mapped_smiles(y)
        if y.hash not in score_cache:
            score_cache[y.hash] = y.bond_mat_scores[0]

    parents = list(seed_objs)
    iter_idx = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        while parents:
            if max_iter is not None and iter_idx >= max_iter:
                break
            iter_idx += 1
            action_label = f"cycle{iter_idx}_b{n_break}_f{n_form}"

            # 1) 断键阶段：并行生成，主进程用 mid_hashes 做跨 parent 去重
            if allow_break and n_break > 0:
                break_results = list(executor.map(_break_bonds_worker, parents, repeat(n_break)))
                intermediates: List[Tuple[yp.yarpecule, yp.yarpecule]] = []
                mid_hashes = set()
                for parent, mids in zip(parents, break_results):
                    added = False
                    if mids:
                        for mid in mids:
                            if mid.hash in mid_hashes:
                                continue
                            mid_hashes.add(mid.hash)
                            intermediates.append((parent, mid))
                            added = True
                    if not mids or not added:
                        intermediates.append((parent, parent))
            else:
                intermediates = [(p, p) for p in parents]

            if not intermediates:
                break

            # 2) 成键阶段：逐步并行，跨 parent 用 step_hashes 去重
            frontier = intermediates
            for _ in range(max(n_form, 0)):
                if not frontier:
                    break
                mols = [mol for _, mol in frontier]
                form_results = list(executor.map(_form_bonds_worker, mols, repeat(inter), repeat(intra)))
                step_hashes = set()
                next_frontier: List[Tuple[yp.yarpecule, yp.yarpecule]] = []
                for (origin, _), prods in zip(frontier, form_results):
                    for prod in prods:
                        if prod.hash in step_hashes:
                            continue
                        step_hashes.add(prod.hash)
                        next_frontier.append((origin, prod))
                frontier = next_frontier

            # 3) 收集本轮结果
            new_parents: List[yp.yarpecule] = []
            for origin, prod in frontier if n_form > 0 else intermediates:
                bmat = prod.bond_mats[0]
                if has_high_order_bond(bmat):
                    continue
                if score_thresh is not None and prod.bond_mat_scores[0] > score_thresh:
                    continue
                fc = yp.return_formals(bmat, prod.elements)
                if any(abs(chg) > 1 for chg in fc):
                    continue
                if sum(1 for chg in fc if chg != 0) > 2:
                    continue
                if origin.hash == prod.hash:
                    continue

                rxn_key = tuple(sorted((origin.hash, prod.hash)))
                if rxn_key in reaction_pairs:
                    continue
                reaction_pairs.add(rxn_key)

                if prod.hash not in product_hashes:
                    product_hashes.add(prod.hash)
                    mapped_cache[prod.hash] = yarpecule_to_mapped_smiles(prod)
                    score_cache[prod.hash] = prod.bond_mat_scores[0]
                    products.append(prod)
                    new_parents.append(prod)
                else:
                    if prod.hash not in mapped_cache:
                        mapped_cache[prod.hash] = yarpecule_to_mapped_smiles(prod)
                    if prod.hash not in score_cache:
                        score_cache[prod.hash] = prod.bond_mat_scores[0]

                edges.append((origin.hash, prod.hash, action_label))

            parents = new_parents

    if logger:
        elapsed = time.time() - mode_start
        logger.log(
            f"[b{n_break}f{n_form}] done | +{len(products)} products | +{len(edges)} edges | workers={num_workers} | elapsed={elapsed:.2f}s"
        )

    return products, edges, mapped_cache, score_cache


def write_products(path: str, products: List[yp.yarpecule], mapped_cache: dict, score_cache: dict) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hash", "mapped_smiles", "score"])
        for p in products:
            writer.writerow([p.hash, mapped_cache[p.hash], score_cache[p.hash]])


def write_edges(path: str, edges: List[Tuple[float, float, str]], mapped_cache: dict, score_cache: dict) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parent_hash", "parent_smiles", "parent_score", "child_hash", "child_smiles", "child_score", "action"])
        for parent, child, action in edges:
            writer.writerow([parent, mapped_cache[parent], score_cache[parent], child, mapped_cache[child], score_cache[child], action])


def build_reaction_modes(n_break: int, n_form: int, max_break: int = None, max_form: int = None) -> List[Tuple[int, int]]:
    """
    返回需要执行的 (n_break, n_form) 组合。
    - 若 max_break/max_form 均未提供，返回单一组合 (n_break, n_form)。
    - 若提供上限，则遍历断键/成键不超过该上限的所有组合，跳过 (0, 0)。
      顺序：先“只断键”、再“只成键”、再“断键+成键”，与示例 b1f0,b0f1,b1f1 一致。
    """
    if max_break is None and max_form is None:
        return [(n_break, n_form)]

    max_b = max_break if max_break is not None else n_break
    max_f = max_form if max_form is not None else n_form
    if max_b < 0 or max_f < 0:
        raise ValueError("max-break 和 max-form 必须为非负整数")

    modes: List[Tuple[int, int]] = []
    for b in range(1, max_b + 1):
        modes.append((b, 0))
    for f in range(1, max_f + 1):
        modes.append((0, f))
    for b in range(1, max_b + 1):
        for f in range(1, max_f + 1):
            modes.append((b, f))

    return modes or [(0, 0)]


def enumerate_reaction_modes(
    seeds: Sequence[yp.yarpecule],
    reaction_modes: Sequence[Tuple[int, int]],
    allow_break: bool = True,
    inter: bool = True,
    intra: bool = True,
    score_thresh: float = None,
    max_depth: int = None,
    logger: Optional[RunLogger] = None,
    num_workers: int = 1,
) -> Tuple[List[yp.yarpecule], List[Tuple[float, float, str]], dict, dict]:
    """
    按反应深度分层：每一层对 frontier 依次执行所有 (n_break, n_form) 组合，每个组合在该层只跑一轮（内部 max_iter=1）。
    跨组合共享反应/产物去重；返回累积的唯一分子、全部去重后的反应边以及缓存。
    num_workers>1 时，单个 (n_break, n_form) 组合内部会按 seed 并行。
    """
    product_hashes: Set[float] = {y.hash for y in seeds}
    reaction_pairs: Set[Tuple[float, float]] = set()
    mapped_cache: Dict[float, str] = {y.hash: yarpecule_to_mapped_smiles(y) for y in seeds}
    score_cache: Dict[float, float] = {y.hash: y.bond_mat_scores[0] for y in seeds}
    products: List[yp.yarpecule] = list(seeds)
    edges: List[Tuple[float, float, str]] = []
    frontier: List[yp.yarpecule] = list(seeds)
    depth = 0

    while frontier and (max_depth is None or depth < max_depth):
        depth += 1
        depth_new: List[yp.yarpecule] = []
        if logger:
            logger.log("-" * 8 + f" depth {depth} " + "-" * 8)
        for n_break, n_form in reaction_modes:
            run_products, run_edges, mapped_cache, score_cache = enumerate_reaction_mode_by_seed(
                frontier,
                n_break=n_break,
                n_form=n_form,
                allow_break=allow_break,
                inter=inter,
                intra=intra,
                score_thresh=score_thresh,
                max_iter=1,  # 每层每个模式只跑一轮
                product_hashes=product_hashes,
                reaction_pairs=reaction_pairs,
                mapped_cache=mapped_cache,
                score_cache=score_cache,
                logger=logger,
                num_workers=num_workers,
            )
            products.extend(run_products)
            edges.extend(run_edges)
            depth_new.extend(run_products)
        frontier = depth_new

    return products, edges, mapped_cache, score_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Reaction enumeration (break then form) with atom-mapped SMILES output.")
    parser.add_argument("seed", nargs="+", help="起始分子（SMILES 或 .xyz/.mol 等文件路径）。")
    parser.add_argument("--n-break", type=int, default=2, help="断键数目（默认 2，0 表示不做断键）。")
    parser.add_argument("--n-form", type=int, default=2, help="成键迭代次数（默认 2）。")
    parser.add_argument("--max-break", type=int, default=None, help="断键数目上限。指定后会遍历所有不超过该值的断键组合（会跳过 0 断键 0 成键）。")
    parser.add_argument("--max-form", type=int, default=None, help="成键迭代次数上限。指定后会遍历所有不超过该值的成键组合（会跳过 0 断键 0 成键）。")
    parser.add_argument("--max-depth", type=int, default=None, help="最大反应深度（每层依次跑所有 b/f 组合），默认直到无新增。")
    parser.add_argument("--no-break", action="store_true", help="禁用断键阶段。")
    parser.add_argument("--no-inter", action="store_true", default=False, help="禁用分子间成键（默认允许）")
    parser.add_argument("--no-intra", action="store_true", default=False, help="禁用分子内成键（默认允许分子内成键）")
    parser.add_argument("--score-thresh", type=float, default=0.0, help="过滤 bond_mat_scores[0] 超过阈值的结构。")
    parser.add_argument("--out-prefix", default="run1", help="输出前缀（默认 run）。")
    parser.add_argument("--log-file", default=None, help="日志文件路径（默认 {out_prefix}.log）。")
    parser.add_argument("--num-workers", type=int, default=1, help="并行核数（默认 1，表示不并行）。并行按 seed 颗粒度。")
    args = parser.parse_args()

    seeds = build_seeds(args.seed)
    reaction_modes = build_reaction_modes(args.n_break, args.n_form, args.max_break, args.max_form)
    log_path = args.log_file or f"{args.out_prefix}.log"
    logger = RunLogger(log_path)
    run_start = time.time()
    try:
        logger.log(f"Start enumeration | seeds={len(seeds)} | modes={', '.join(f'b{b}f{f}' for b, f in reaction_modes)} | allow_break={not args.no_break} inter={not args.no_inter} intra={not args.no_intra} score_thresh={args.score_thresh} max_depth={args.max_depth}")

        products, edges, mapped_cache, score_cache = enumerate_reaction_modes(
            seeds,
            reaction_modes,
            allow_break=not args.no_break,
            inter=not args.no_inter,
            intra=not args.no_intra,
            score_thresh=args.score_thresh,
            max_depth=args.max_depth,
            logger=logger,
            num_workers=args.num_workers,
        )

        prod_path = f"{args.out_prefix}_products.csv"
        edge_path = f"{args.out_prefix}_edges.csv"
        write_products(prod_path, products, mapped_cache, score_cache)
        write_edges(edge_path, edges, mapped_cache, score_cache)

        elapsed = time.time() - run_start
        logger.log(f"Done | Seeds: {len(seeds)} | Unique structures: {len(products)} | Edges: {len(edges)} | Elapsed: {elapsed:.2f}s")
        if reaction_modes:
            mode_str = ", ".join(f"b{b}f{f}" for b, f in reaction_modes)
            logger.log(f"Reaction modes: {mode_str}")
        logger.log(f"Products saved to: {prod_path}")
        logger.log(f"Edges saved to:    {edge_path}")
        logger.log(f"Log saved to:      {log_path}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
