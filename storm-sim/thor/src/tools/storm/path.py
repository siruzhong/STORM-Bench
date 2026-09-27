"""Loop-path planning for room rollouts.

Default rollout pipeline (adapted from Qwen-RobotNav's ObjectNav generation):
  reachable mask -> medial-axis main component -> inward closed contour
  -> collision-checked spline -> curvature-aware timing -> (x, z, body yaw)

Exploration pipeline:
  medial-axis skeleton -> pruned DFS traversal with dead-end backtracking
  -> cubic-spline smoothing -> true arc-length interpolation

Fallback pipeline:
  reachable points -> perimeter anchors -> reachable-grid shortest paths
  -> true arc-length interpolation -> per-frame (x, z, yaw)

Legacy spline pipeline:
  reachable points -> convex hull (meter inset) -> periodic cubic B-spline
  -> dense evaluation -> true arc-length interpolation -> per-frame (x, z, yaw)
"""
from __future__ import annotations

import heapq
import math
import random
from collections import deque

import numpy as np
from scipy import ndimage
from scipy.interpolate import splev, splprep
from scipy.spatial import ConvexHull, QhullError, cKDTree
from skimage.measure import find_contours
from skimage.morphology import skeletonize

DEFAULT_TOUR_OBJECT_TYPES = (
    "CounterTop",
    "Sink",
    "StoveBurner",
    "Fridge",
    "DiningTable",
    "CoffeeTable",
    "Sofa",
    "Bed",
    "Desk",
    "TVStand",
    "Dresser",
    "SideTable",
)

DEFAULT_LOOK_OBJECT_WEIGHTS = {
    "CounterTop": 2.0,
    "Sink": 1.8,
    "StoveBurner": 1.8,
    "Fridge": 1.5,
    "DiningTable": 1.5,
    "CoffeeTable": 1.4,
    "Sofa": 1.1,
    "Bed": 1.1,
    "Desk": 1.3,
    "TVStand": 1.1,
    "Dresser": 1.0,
    "SideTable": 1.0,
    "Cabinet": 1.25,
    "Shelf": 1.15,
    "ShelvingUnit": 1.15,
    "Microwave": 1.25,
    "CoffeeMachine": 1.0,
    "Toaster": 1.0,
    "GarbageCan": 0.8,
    "Window": 0.8,
    "HousePlant": 0.8,
}

FUNCTIONAL_AREA_OBJECT_TYPES = frozenset({
    *DEFAULT_TOUR_OBJECT_TYPES,
    "Cabinet",
    "Shelf",
    "ShelvingUnit",
    "Microwave",
    "CoffeeMachine",
    "Toaster",
    "GarbageCan",
})


def _functional_area_groups(objects, merge_distance=1.25):
    """Group nearby fixed landmarks into distinct functional areas."""
    landmarks = [
        obj for obj in objects or []
        if obj.get("objectType") in FUNCTIONAL_AREA_OBJECT_TYPES
        and obj.get("position")
    ]
    if not landmarks:
        return []

    points = np.array([_position_xz(obj) for obj in landmarks], dtype=float)
    parent = list(range(len(points)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in cKDTree(points).query_pairs(max(float(merge_distance), 0.0)):
        union(left, right)

    grouped = {}
    for index, obj in enumerate(landmarks):
        grouped.setdefault(find(index), []).append(obj)
    return [grouped[key] for key in sorted(grouped)]


def functional_route_coverage(route, objects, max_distance=1.6,
                              merge_distance=1.25):
    """Measure how many landmark-defined functional areas a route enters."""
    groups = _functional_area_groups(objects, merge_distance=merge_distance)
    route_points = []
    for point in route:
        if isinstance(point, dict):
            route_points.append(_position_xz(point))
        else:
            route_points.append(np.asarray(point[:2], dtype=float))

    if not groups:
        return {
            "functional_zone_count": 0,
            "functional_zone_covered": 0,
            "functional_zone_coverage": 1.0,
            "functional_zone_distance_mean_m": 0.0,
            "functional_zone_distance_max_m": 0.0,
        }
    if not route_points:
        return {
            "functional_zone_count": len(groups),
            "functional_zone_covered": 0,
            "functional_zone_coverage": 0.0,
            "functional_zone_distance_mean_m": float("inf"),
            "functional_zone_distance_max_m": float("inf"),
        }

    route_tree = cKDTree(np.asarray(route_points, dtype=float))
    distances = []
    for group in groups:
        landmarks = np.array([_position_xz(obj) for obj in group], dtype=float)
        distances.append(float(np.min(route_tree.query(landmarks, k=1)[0])))
    covered = sum(distance <= float(max_distance) for distance in distances)
    return {
        "functional_zone_count": len(groups),
        "functional_zone_covered": covered,
        "functional_zone_coverage": covered / len(groups),
        "functional_zone_distance_mean_m": float(np.mean(distances)),
        "functional_zone_distance_max_m": float(np.max(distances)),
    }


def functional_visibility_coverage(visible_object_ids, objects,
                                   merge_distance=1.25):
    """Measure functional-area coverage using actual rendered visibility."""
    groups = _functional_area_groups(objects, merge_distance=merge_distance)
    visible = set(visible_object_ids or [])
    covered = sum(
        any(obj.get("objectId") in visible for obj in group)
        for group in groups
    )
    return {
        "functional_zone_visible": covered,
        "functional_zone_visible_coverage": (
            covered / len(groups) if groups else 1.0
        ),
    }


def loop_ring(positions, inset: float = 0.4):
    """Ordered, closed ring of (x, z) hull waypoints, inset by meters.

    `inset` is treated as a metric distance toward the room centroid. The older
    implementation used it as a scale fraction, which pulled large rooms too far
    inward and made the camera path feel cramped.
    """
    pts = np.array([[p["x"], p["z"]] for p in positions])
    hull = ConvexHull(pts)
    ring = pts[hull.vertices]
    centroid = ring.mean(axis=0)
    vec = ring - centroid
    dist = np.linalg.norm(vec, axis=1, keepdims=True)
    scale = np.maximum((dist - inset) / np.maximum(dist, 1e-6), 0.05)
    ring = centroid + vec * scale
    return np.vstack([ring, ring[0]])  # closed


def spline_loop(positions, inset: float = 0.4, smooth: float = 0.0, dense: int = 6000):
    """Fit a periodic cubic B-spline through the inset hull points.

    Returns (xy, cum_len) where xy is (dense, 2) points sampled uniformly in the
    spline parameter and cum_len is the cumulative arc length along them. `smooth`
    is scipy's `s`; 0 interpolates the control points, larger values relax toward
    a rounder loop.
    """
    ring = loop_ring(positions, inset=inset)
    ctrl = ring[:-1]  # drop duplicated closing point; per=1 closes it
    x, z = ctrl[:, 0], ctrl[:, 1]
    # k=3 cubic if enough points, else fall back to the polygon degree.
    k = 3 if len(ctrl) > 3 else max(1, len(ctrl) - 1)
    tck, _ = splprep([x, z], s=smooth, per=1, k=k)
    u = np.linspace(0.0, 1.0, dense, endpoint=False)
    xs, zs = splev(u, tck)
    xy = np.column_stack([xs, zs])
    seg = np.hypot(np.diff(xy[:, 0], append=xy[0, 0]),
                   np.diff(xy[:, 1], append=xy[0, 1]))
    cum_len = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    return xy, cum_len


def heading_to(dx: float, dz: float) -> float:
    """AI2-THOR yaw: 0 = +z, increasing clockwise toward +x."""
    return math.degrees(math.atan2(dx, dz)) % 360.0


def _interp_closed(xy, cum_len, distance):
    """Interpolate a closed polyline at an arc-length distance."""
    total = cum_len[-1] + float(np.linalg.norm(xy[0] - xy[-1]))
    d = distance % total
    closed_len = np.append(cum_len, total)
    closed_xy = np.vstack([xy, xy[0]])
    x = np.interp(d, closed_len, closed_xy[:, 0])
    z = np.interp(d, closed_len, closed_xy[:, 1])
    return np.array([x, z]), total


def _infer_grid_step(positions) -> float:
    vals = []
    for axis in ("x", "z"):
        coords = sorted({round(float(p[axis]), 4) for p in positions})
        vals.extend(
            coords[i + 1] - coords[i]
            for i in range(len(coords) - 1)
            if coords[i + 1] - coords[i] > 1e-4
        )
    return min(vals) if vals else 0.25


def _reachable_grid(positions):
    step = _infer_grid_step(positions)
    key_to_point = {}
    for p in positions:
        key = (round(float(p["x"]) / step), round(float(p["z"]) / step))
        key_to_point[key] = np.array([float(p["x"]), float(p["z"])])
    return step, key_to_point


def _point_key(point, step):
    return (round(float(point["x"]) / step), round(float(point["z"]) / step))


def _reachable_context(positions):
    """Grid metadata used to avoid wall-hugging and dead-end viewpoints."""
    step, key_to_point = _reachable_grid(positions)
    keys = set(key_to_point)
    neighbor_count = {key: sum(1 for _ in _grid_neighbors(key, keys)) for key in keys}
    boundary = {key for key, count in neighbor_count.items() if count < 4}

    clearance_steps = {key: 0 for key in boundary}
    queue = deque(boundary)
    while queue:
        key = queue.popleft()
        for nb in _grid_neighbors(key, keys):
            if nb in clearance_steps:
                continue
            clearance_steps[nb] = clearance_steps[key] + 1
            queue.append(nb)

    pts = np.array(list(key_to_point.values()))
    return {
        "step": step,
        "key_to_point": key_to_point,
        "keys": keys,
        "neighbor_count": neighbor_count,
        "clearance": {
            key: clearance_steps.get(key, 0) * step
            for key in keys
        },
        "center": pts.mean(axis=0),
    }


def _grid_neighbors(key, keys):
    x, z = key
    for nb in ((x + 1, z), (x - 1, z), (x, z + 1), (x, z - 1)):
        if nb in keys:
            yield nb


def _largest_mask_component(mask):
    """Keep the largest 4-connected navigable region."""
    structure = ndimage.generate_binary_structure(2, 1)
    labels, count = ndimage.label(mask, structure=structure)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def _reachable_mask(key_to_point):
    """Rasterize reachable-grid keys into a compact boolean occupancy mask."""
    keys = set(key_to_point)
    min_x = min(k[0] for k in keys)
    max_x = max(k[0] for k in keys)
    min_z = min(k[1] for k in keys)
    max_z = max(k[1] for k in keys)
    mask = np.zeros((max_z - min_z + 1, max_x - min_x + 1), dtype=bool)
    for x, z in keys:
        mask[z - min_z, x - min_x] = True
    return mask, (min_x, min_z)


def _skeleton_graph(key_to_point):
    """Build an 8-connected medial-axis graph from the reachable mask."""
    mask, (min_x, min_z) = _reachable_mask(key_to_point)
    medial = skeletonize(_largest_mask_component(mask))
    nodes = {
        (int(col + min_x), int(row + min_z))
        for row, col in np.argwhere(medial)
    }
    graph = {}
    offsets = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )
    for x, z in nodes:
        graph[(x, z)] = {
            (x + dx, z + dz)
            for dx, dz in offsets
            if (x + dx, z + dz) in nodes
        }
    return graph


def _largest_graph_component(graph):
    remaining = set(graph)
    largest = set()
    while remaining:
        start = min(remaining)
        component = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nb in graph[node]:
                if nb not in component:
                    component.add(nb)
                    queue.append(nb)
        remaining -= component
        if len(component) > len(largest):
            largest = component
    return {node: graph[node] & largest for node in largest}


def _prune_short_skeleton_branches(graph, max_branch_steps):
    """Remove endpoint spurs that terminate quickly at a junction."""
    graph = {node: set(neighbors) for node, neighbors in graph.items()}
    max_branch_steps = max(0, int(max_branch_steps))
    if max_branch_steps == 0:
        return _largest_graph_component(graph)

    while True:
        remove = set()
        for endpoint in sorted(node for node, nbs in graph.items() if len(nbs) == 1):
            path = [endpoint]
            previous = None
            current = endpoint
            while len(path) - 1 <= max_branch_steps:
                forward = [nb for nb in graph[current] if nb != previous]
                if not forward:
                    break
                nxt = forward[0]
                path.append(nxt)
                previous, current = current, nxt
                if len(graph[current]) != 2:
                    break
            branch_steps = len(path) - 1
            if len(graph[current]) >= 3 and branch_steps <= max_branch_steps:
                remove.update(path[:-1])
        if not remove:
            break
        for node in remove:
            for nb in graph.get(node, ()):
                graph[nb].discard(node)
            graph.pop(node, None)
    return _largest_graph_component(graph)


def _graph_shortest_path(graph, start, goal):
    """Shortest medial-axis path, with metric cost for diagonal edges."""
    frontier = [(0.0, start)]
    cost = {start: 0.0}
    parent = {start: None}
    while frontier:
        current_cost, node = heapq.heappop(frontier)
        if node == goal:
            break
        if current_cost > cost.get(node, float("inf")):
            continue
        for nb in graph[node]:
            edge_cost = math.hypot(nb[0] - node[0], nb[1] - node[1])
            candidate = current_cost + edge_cost
            if candidate < cost.get(nb, float("inf")):
                cost[nb] = candidate
                parent[nb] = node
                heapq.heappush(frontier, (candidate, nb))
    if goal not in parent:
        raise ValueError(f"no skeleton path between {start} and {goal}")
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return list(reversed(path))


def _explore_skeleton(graph, seed, coverage):
    """Random DFS exploration with dead-end backtracking, closed at the start."""
    rng = random.Random(int(seed))
    start_candidates = sorted(node for node, nbs in graph.items() if len(nbs) >= 2)
    start = rng.choice(start_candidates or sorted(graph))
    target_count = min(
        len(graph),
        max(2, int(math.ceil(len(graph) * min(max(float(coverage), 0.05), 1.0)))),
    )

    route = [start]
    visited = {start}
    stack = [start]
    neighbor_orders = {}
    while stack and len(visited) < target_count:
        node = stack[-1]
        if node not in neighbor_orders:
            options = sorted(graph[node])
            rng.shuffle(options)
            neighbor_orders[node] = options
        options = [nb for nb in neighbor_orders[node] if nb not in visited]
        if options:
            nxt = options[0]
            visited.add(nxt)
            stack.append(nxt)
            route.append(nxt)
            continue
        stack.pop()
        if stack:
            route.append(stack[-1])

    if route[-1] != start:
        route.extend(_graph_shortest_path(graph, route[-1], start)[1:])
    return route


def _expand_diagonal_steps(route, ctx):
    """Replace diagonal skeleton edges with collision-safe reachable-grid steps."""
    expanded = [route[0]]
    for goal in route[1:]:
        start = expanded[-1]
        dx, dz = goal[0] - start[0], goal[1] - start[1]
        if abs(dx) == 1 and abs(dz) == 1:
            candidates = [
                (goal[0], start[1]),
                (start[0], goal[1]),
            ]
            candidates = [key for key in candidates if key in ctx["keys"]]
            if candidates:
                middle = max(
                    candidates,
                    key=lambda key: (ctx["clearance"].get(key, 0.0), key),
                )
                if middle != expanded[-1]:
                    expanded.append(middle)
        if goal != expanded[-1]:
            expanded.append(goal)
    return expanded


def _smooth_skeleton_route(route, ctx, smoothing):
    """Fit a closed cubic spline and reject curves leaving navigable space."""
    route = _expand_diagonal_steps(route, ctx)
    control = np.array([ctx["key_to_point"][key] for key in route], dtype=float)
    control, cumulative = _polyline_cumulative(control)
    total = float(cumulative[-1])
    if len(control) < 4 or total <= ctx["step"]:
        return control

    # The skeleton already supplies regular 0.25 m controls. Try the requested
    # smoothing first, then progressively tighten before falling back to it.
    query_count = max(100, int(math.ceil(total / max(ctx["step"] / 8.0, 0.01))))
    u = cumulative / total
    reachable_tree = cKDTree(np.array(list(ctx["key_to_point"].values())))
    max_grid_distance = ctx["step"] * 0.8
    base_s = len(control) * max(float(smoothing), 0.0) ** 2
    for smooth_score in (base_s, base_s * 0.25, 0.0):
        try:
            tck, _ = splprep(
                [control[:, 0], control[:, 1]],
                u=u,
                s=smooth_score,
                per=True,
                k=min(3, len(control) - 1),
            )
            sampled_u = np.linspace(0.0, 1.0, query_count, endpoint=True)
            xs, zs = splev(sampled_u, tck)
            curve = np.column_stack([xs, zs])
        except (TypeError, ValueError):
            continue
        distances, _ = reachable_tree.query(curve, k=1)
        if float(np.max(distances)) <= max_grid_distance:
            curve[-1] = curve[0]
            return curve
    return control


def skeleton_exploration_polyline(positions, seed=0, coverage=0.65,
                                  prune_length=0.5, smoothing=0.05):
    """Create one closed, centreline-biased exploration route.

    This follows the trajectory construction in Qwen-RobotNav section 4.1.3,
    adapted to a repeatable rollout by closing the explored path back to its
    starting point.
    """
    ctx = _reachable_context(positions)
    graph = _skeleton_graph(ctx["key_to_point"])
    prune_steps = int(round(max(float(prune_length), 0.0) / ctx["step"]))
    graph = _prune_short_skeleton_branches(graph, prune_steps)
    if len(graph) < 2:
        raise ValueError("navigable skeleton is too small to build an exploration path")
    route = _explore_skeleton(graph, seed=seed, coverage=coverage)
    return _smooth_skeleton_route(route, ctx, smoothing=smoothing)


def _contour_perimeter(contour):
    if len(contour) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(contour, axis=0), axis=1).sum())


def _contour_area(contour):
    if len(contour) < 3:
        return 0.0
    x = contour[:, 0]
    z = contour[:, 1]
    return 0.5 * float(np.sum(x[:-1] * z[1:] - x[1:] * z[:-1]))


def _smooth_closed_contour(contour, ctx, smoothing):
    """Round a closed free-space contour without leaving navigable space."""
    contour = np.asarray(contour, dtype=float)
    if float(np.linalg.norm(contour[0] - contour[-1])) > 1e-6:
        contour = np.vstack([contour, contour[0]])
    contour, cumulative = _polyline_cumulative(contour)
    total = float(cumulative[-1])
    if len(contour) < 5 or total <= ctx["step"] * 8:
        return contour

    controls = contour
    control_cumulative = cumulative
    control_total = total

    # `splprep(per=True)` still needs the repeated endpoint at u=1. Dropping it
    # leaves the fitted parameter range short of one full revolution and creates
    # a large artificial closing segment at the lap seam.
    u = control_cumulative / control_total
    reachable = np.array(list(ctx["key_to_point"].values()))
    reachable_tree = cKDTree(reachable)
    query_count = max(200, int(math.ceil(total / max(ctx["step"] / 10.0, 0.01))))
    base_s = len(controls) * max(float(smoothing), 0.0) ** 2
    for smooth_score in (base_s, base_s * 0.25, 0.0):
        try:
            tck, _ = splprep(
                [controls[:, 0], controls[:, 1]],
                u=u,
                s=smooth_score,
                per=True,
                k=min(3, len(controls) - 1),
            )
            xs, zs = splev(np.linspace(0.0, 1.0, query_count, endpoint=False), tck)
            curve = np.column_stack([xs, zs])
        except (TypeError, ValueError):
            continue
        distances, _ = reachable_tree.query(curve, k=1)
        if float(np.max(distances)) <= ctx["step"] * 0.8:
            return np.vstack([curve, curve[0]])

    return contour


def skeleton_navigation_loop_polyline(positions, objects=None, seed=0,
                                      clearance=0.5, smoothing=0.05,
                                      functional_zone_max_distance=1.6,
                                      functional_zone_merge_distance=1.25):
    """Build a one-way closed loop suitable for repeated natural rollouts.

    Qwen-RobotNav uses the medial-axis skeleton for exploration trajectories.
    A DFS traversal of that skeleton legitimately backtracks, but replaying the
    traversal as a loop forces repeated 180-degree turns. For a cyclic rollout,
    we instead use the skeleton to identify the main navigable component and
    trace an inward free-space contour around it. The result has no retraced
    edges or dead-end reversals and can be replayed for multiple identical laps.
    """
    ctx = _reachable_context(positions)
    mask, (min_x, min_z) = _reachable_mask(ctx["key_to_point"])
    main_mask = _largest_mask_component(mask)

    # Keep the contour away from walls. If erosion disconnects or collapses a
    # small room, progressively relax it rather than returning a dead-end path.
    requested = max(1, int(round(max(float(clearance), 0.0) / ctx["step"])))
    erosion_options = list(range(requested, 0, -1))
    candidates_by_erosion = {}
    structure = ndimage.generate_binary_structure(2, 1)
    skeleton_graph = _largest_graph_component(_skeleton_graph(ctx["key_to_point"]))
    skeleton_points = np.array(
        [ctx["key_to_point"][key] for key in skeleton_graph], dtype=float)
    functional_groups = _functional_area_groups(
        objects, merge_distance=functional_zone_merge_distance)

    for erosion_steps in erosion_options:
        eroded = (
            ndimage.binary_erosion(main_mask, structure=structure,
                                   iterations=erosion_steps)
            if erosion_steps else main_mask
        )
        if int(eroded.sum()) < 6:
            continue
        labels, component_count = ndimage.label(eroded, structure=structure)
        level_candidates = []
        for component_label in range(1, component_count + 1):
            component = labels == component_label
            if int(component.sum()) < 6:
                continue
            padded = np.pad(component.astype(float), 1)
            for raw in find_contours(padded, 0.5):
                if len(raw) < 10 or float(np.linalg.norm(raw[0] - raw[-1])) > 1e-6:
                    continue
                curve = np.column_stack([
                    (raw[:, 1] - 1.0 + min_x) * ctx["step"],
                    (raw[:, 0] - 1.0 + min_z) * ctx["step"],
                ])
                perimeter = _contour_perimeter(curve)
                area = abs(_contour_area(curve))
                if perimeter < max(2.5, ctx["step"] * 12) or area < ctx["step"] ** 2 * 5:
                    continue
                curve_tree = cKDTree(curve)
                if len(skeleton_points):
                    covered = curve_tree.query(skeleton_points, k=1)[0]
                    skeleton_coverage = float(np.mean(
                        covered <= max(erosion_steps * ctx["step"] * 3.0, 1.0)))
                else:
                    skeleton_coverage = 0.0
                if functional_groups:
                    zone_distances = []
                    for group in functional_groups:
                        landmarks = np.array(
                            [_position_xz(obj) for obj in group], dtype=float)
                        zone_distances.append(float(np.min(
                            curve_tree.query(landmarks, k=1)[0])))
                    functional_coverage = float(np.mean(
                        np.asarray(zone_distances) <=
                        float(functional_zone_max_distance)))
                else:
                    functional_coverage = 1.0
                score = perimeter * (
                    0.35 + 0.20 * skeleton_coverage + 0.45 * functional_coverage)
                level_candidates.append({
                    "score": score,
                    "erosion_steps": erosion_steps,
                    "area": area,
                    "functional_coverage": functional_coverage,
                    "curve": curve,
                })
        if level_candidates:
            candidates_by_erosion[erosion_steps] = max(
                level_candidates,
                key=lambda item: (
                    item["functional_coverage"], item["score"], item["area"]),
            )

    # Retain the requested wall clearance unless relaxing one grid cell yields a
    # substantially broader route. This catches narrow doorways without making
    # ordinary scenes hug the walls.
    selected = None
    for erosion_steps in erosion_options:
        candidate = candidates_by_erosion.get(erosion_steps)
        if candidate is None:
            continue
        if selected is None:
            selected = candidate
            continue
        coverage_gain = (
            candidate["functional_coverage"] -
            selected["functional_coverage"]
        )
        if coverage_gain > 1e-9 or (
                abs(coverage_gain) <= 1e-9
                and candidate["score"] > selected["score"] * 1.45):
            selected = candidate
    if selected is None:
        raise ValueError("navigable component has no closed walking loop")

    rng = random.Random(int(seed))
    curve = selected["curve"]

    # Seed controls traversal direction and seam location without changing the
    # geometry. This keeps dataset episodes deterministic but avoids a fixed
    # first frame across seeds.
    curve = curve[:-1]
    if rng.random() < 0.5:
        curve = curve[::-1]
    offset = rng.randrange(len(curve))
    curve = np.roll(curve, -offset, axis=0)
    curve = np.vstack([curve, curve[0]])
    return _smooth_closed_contour(curve, ctx, smoothing=smoothing)


def _surface_footprint_distance(point, surface):
    """XZ distance from a point to a surface's axis-aligned footprint."""
    box = surface.get("axisAlignedBoundingBox") or {}
    center = box.get("center") or surface.get("position") or {}
    size = box.get("size") or {}
    center_x = float(center.get("x", 0.0))
    center_z = float(center.get("z", 0.0))
    half_x = max(float(size.get("x", 0.0)) / 2.0, 0.0)
    half_z = max(float(size.get("z", 0.0)) / 2.0, 0.0)
    dx = max(abs(float(point[0]) - center_x) - half_x, 0.0)
    dz = max(abs(float(point[1]) - center_z) - half_z, 0.0)
    return math.hypot(dx, dz)


def _surface_footprint_hull(surfaces):
    """Return convex-hull equations for all functional-surface footprints."""
    corners = []
    for surface in surfaces:
        box = surface.get("axisAlignedBoundingBox") or {}
        center = box.get("center") or surface.get("position") or {}
        size = box.get("size") or {}
        center_x = float(center.get("x", 0.0))
        center_z = float(center.get("z", 0.0))
        half_x = max(float(size.get("x", 0.0)) / 2.0, 0.01)
        half_z = max(float(size.get("z", 0.0)) / 2.0, 0.01)
        corners.extend((
            (center_x - half_x, center_z - half_z),
            (center_x - half_x, center_z + half_z),
            (center_x + half_x, center_z - half_z),
            (center_x + half_x, center_z + half_z),
        ))
    points = np.unique(np.asarray(corners, dtype=float), axis=0)
    if len(points) < 3:
        raise ValueError("functional surfaces cannot define an interior hull")
    try:
        return ConvexHull(points)
    except QhullError as exc:
        raise ValueError(
            "functional surfaces cannot define a stable interior hull") from exc


def _inside_surface_hull(point, hull, margin=0.0):
    point = np.asarray(point, dtype=float)
    values = hull.equations[:, :-1] @ point + hull.equations[:, -1]
    return bool(np.max(values) <= float(margin) + 1e-9)


def _graph_farthest(graph, start):
    distance = {start: 0}
    frontier = deque([start])
    while frontier:
        node = frontier.popleft()
        for neighbor in graph[node]:
            if neighbor in distance:
                continue
            distance[neighbor] = distance[node] + 1
            frontier.append(neighbor)
    return max(distance, key=lambda node: (distance[node], node))


def _graph_components(graph):
    """Return deterministic connected components for an adjacency mapping."""
    components = []
    remaining = set(graph)
    while remaining:
        root = min(remaining)
        component = {root}
        frontier = [root]
        while frontier:
            node = frontier.pop()
            for neighbor in graph[node]:
                if neighbor not in component:
                    component.add(neighbor)
                    frontier.append(neighbor)
        remaining -= component
        components.append(component)
    return components


def _skeletonize_grid_keys(keys):
    """Thin a set of reachable-grid keys without changing grid coordinates."""
    if not keys:
        return set()
    min_x = min(key[0] for key in keys)
    max_x = max(key[0] for key in keys)
    min_z = min(key[1] for key in keys)
    max_z = max(key[1] for key in keys)
    mask = np.zeros((max_z - min_z + 1, max_x - min_x + 1), dtype=bool)
    for x, z in keys:
        mask[z - min_z, x - min_x] = True
    thinned = skeletonize(mask)
    return {
        (int(column + min_x), int(row + min_z))
        for row, column in np.argwhere(thinned)
    }


def _band_skeleton_graph(keys, traversable_keys):
    """Build an 8-neighbor skeleton graph with collision-safe diagonals."""
    graph = {key: set() for key in keys}
    offsets = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )
    for x, z in keys:
        for dx, dz in offsets:
            neighbor = (x + dx, z + dz)
            if neighbor not in keys:
                continue
            if dx and dz:
                bridges = ((x + dx, z), (x, z + dz))
                if not any(bridge in traversable_keys for bridge in bridges):
                    continue
            graph[(x, z)].add(neighbor)
    return {key: neighbors for key, neighbors in graph.items() if neighbors}


def _turn_angle(previous, current, following):
    incoming = np.asarray(current, dtype=float) - np.asarray(previous, dtype=float)
    outgoing = np.asarray(following, dtype=float) - np.asarray(current, dtype=float)
    denominator = float(np.linalg.norm(incoming) * np.linalg.norm(outgoing))
    if denominator <= 1e-12:
        return 0.0
    cosine = float(np.clip(np.dot(incoming, outgoing) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _weighted_graph_path(graph, start, goal, node_distances,
                         preferred_standoff, tolerance, turn_cost=0.08,
                         standoff_cost=0.60):
    """Shortest path with stable standoff and low-turn continuation costs."""
    if start == goal:
        return [start]
    serial = 0
    initial = (None, start)
    frontier = [(0.0, serial, initial)]
    costs = {initial: 0.0}
    parents = {initial: None}
    final_state = None
    scale = max(float(tolerance), 1e-3)
    while frontier:
        current_cost, _, state = heapq.heappop(frontier)
        previous, current = state
        if current_cost > costs.get(state, float("inf")):
            continue
        if current == goal:
            final_state = state
            break
        for neighbor in sorted(graph[current]):
            edge_length = math.hypot(
                neighbor[0] - current[0], neighbor[1] - current[1])
            distance_error = abs(
                float(node_distances[neighbor]) - float(preferred_standoff))
            penalty = edge_length * float(standoff_cost) * (distance_error / scale) ** 2
            if previous is not None:
                angle = _turn_angle(previous, current, neighbor)
                penalty += float(turn_cost) * (angle / 90.0) ** 2
            candidate = current_cost + edge_length + penalty
            next_state = (current, neighbor)
            if candidate + 1e-12 >= costs.get(next_state, float("inf")):
                continue
            costs[next_state] = candidate
            parents[next_state] = state
            serial += 1
            heapq.heappush(frontier, (candidate, serial, next_state))
    if final_state is None:
        raise ValueError(f"no weighted graph path between {start} and {goal}")
    path = []
    state = final_state
    while state is not None:
        path.append(state[1])
        state = parents[state]
    return list(reversed(path))


def _expand_graph_diagonals(route, allowed_keys, node_distances,
                            preferred_standoff):
    """Expand diagonal skeleton edges to orthogonal reachable-grid steps."""
    expanded = [route[0]]
    for goal in route[1:]:
        start = expanded[-1]
        dx, dz = goal[0] - start[0], goal[1] - start[1]
        if abs(dx) == 1 and abs(dz) == 1:
            bridges = [
                key for key in ((goal[0], start[1]), (start[0], goal[1]))
                if key in allowed_keys
            ]
            if not bridges:
                raise ValueError("skeleton diagonal has no reachable grid bridge")
            middle = min(
                bridges,
                key=lambda key: (
                    abs(node_distances[key] - float(preferred_standoff)), key),
            )
            if middle != expanded[-1]:
                expanded.append(middle)
        if goal != expanded[-1]:
            expanded.append(goal)
    return expanded


def _rounded_closed_polyline(points, radius=0.12):
    """Round every corner of a simple closed grid path."""
    points = np.asarray(points, dtype=float)
    if len(points) > 1 and np.linalg.norm(points[0] - points[-1]) <= 1e-9:
        points = points[:-1]
    if len(points) < 3:
        return np.vstack([points, points[0]])
    rounded = []
    count = len(points)
    for index, corner in enumerate(points):
        previous = points[(index - 1) % count]
        following = points[(index + 1) % count]
        incoming = previous - corner
        outgoing = following - corner
        incoming_length = float(np.linalg.norm(incoming))
        outgoing_length = float(np.linalg.norm(outgoing))
        if incoming_length <= 1e-6 or outgoing_length <= 1e-6:
            rounded.append(corner)
            continue
        incoming /= incoming_length
        outgoing /= outgoing_length
        corner_radius = min(
            float(radius), incoming_length * 0.40, outgoing_length * 0.40)
        entry = corner + incoming * corner_radius
        exit_point = corner + outgoing * corner_radius
        rounded.append(entry)
        for sample in range(1, 5):
            progress = sample / 5.0
            inverse = 1.0 - progress
            rounded.append(
                inverse * inverse * entry
                + 2.0 * inverse * progress * corner
                + progress * progress * exit_point
            )
        rounded.append(exit_point)
    rounded.append(rounded[0])
    return np.asarray(rounded, dtype=float)


def _polyline_length(points):
    points = np.asarray(points, dtype=float)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _surface_coverage_for_keys(keys, key_to_point, surfaces, max_standoff):
    points = [key_to_point[key] for key in keys]
    return sum(
        min(_surface_footprint_distance(point, surface) for point in points)
        <= float(max_standoff)
        for surface in surfaces
    )


def _surface_group_coverage_for_keys(keys, key_to_point, surface_groups,
                                     max_standoff):
    points = [key_to_point[key] for key in keys]
    return sum(
        min(
            _surface_footprint_distance(point, surface)
            for point in points for surface in group
        ) <= float(max_standoff)
        for group in surface_groups
    )


def _alternate_band_return(component_graph, primary, node_distances,
                           preferred_standoff, tolerance,
                           separation_steps, minimum_length_ratio,
                           turn_cost, standoff_cost):
    """Find a disjoint return lane through the wider task-area graph."""
    if len(primary) < 3:
        return None, None
    primary_length = sum(
        math.hypot(right[0] - left[0], right[1] - left[1])
        for left, right in zip(primary, primary[1:])
    )
    for radius in range(max(0, int(separation_steps)), -1, -1):
        blocked = set(primary[1:-1])
        if radius:
            guard = radius + 2
            for x, z in primary[guard:-guard]:
                for dx in range(-radius, radius + 1):
                    remaining = radius - abs(dx)
                    for dz in range(-remaining, remaining + 1):
                        blocked.add((x + dx, z + dz))
        blocked.discard(primary[0])
        blocked.discard(primary[-1])
        return_graph = {
            key: {neighbor for neighbor in neighbors if neighbor not in blocked}
            for key, neighbors in component_graph.items()
            if key not in blocked
        }
        return_graph = {
            key: neighbors for key, neighbors in return_graph.items()
            if neighbors or key in (primary[0], primary[-1])
        }
        if primary[0] not in return_graph or primary[-1] not in return_graph:
            continue
        try:
            route = _weighted_graph_path(
                return_graph,
                primary[-1],
                primary[0],
                node_distances,
                preferred_standoff,
                tolerance,
                turn_cost=turn_cost,
                standoff_cost=standoff_cost,
            )
        except ValueError:
            continue
        return_length = sum(
            math.hypot(right[0] - left[0], right[1] - left[1])
            for left, right in zip(route, route[1:])
        )
        if return_length >= primary_length * float(minimum_length_ratio):
            return route, radius
    return None, None


def _open_route_extension(endpoint, previous, allowed_keys, blocked_keys,
                          node_distances, preferred_standoff, target_steps,
                          beam_width=32):
    """Extend an open route endpoint without revisiting its existing nodes."""
    target_steps = max(0, int(target_steps))
    if target_steps == 0:
        return [endpoint]
    states = [([endpoint], 0.0)]
    best = states[0]
    for _ in range(target_steps):
        expanded = []
        for path, score in states:
            current = path[-1]
            prior = previous if len(path) == 1 else path[-2]
            for neighbor in _grid_neighbors(current, allowed_keys):
                if neighbor in blocked_keys or neighbor in path:
                    continue
                angle = _turn_angle(prior, current, neighbor)
                standoff_error = abs(
                    node_distances[neighbor] - float(preferred_standoff))
                candidate_score = (
                    score
                    + (angle / 90.0) ** 2
                    + 0.20 * standoff_error ** 2
                )
                expanded.append(([*path, neighbor], candidate_score))
        if not expanded:
            break
        expanded.sort(key=lambda item: (item[1], item[0]))
        states = expanded[:max(1, int(beam_width))]
        best = min(states, key=lambda item: (item[1], item[0]))
    return best[0]


def worktop_edge_graph_polyline(
        positions, surfaces, seed=0, preferred_standoff=0.80,
        standoff_tolerance=0.22, min_standoff=0.40, max_standoff=1.60,
        max_spur_length=0.50, return_separation=0.25,
        min_return_length_ratio=0.55, min_length=4.0,
        rounding_radius=0.12, turn_cost=0.08, standoff_cost=0.60,
        surface_merge_distance=1.25, open_extension=0.0,
        extension_max_standoff=2.25, extension_beam_width=32,
        require_closed=False, workspace_interior_only=False,
        interior_hull_margin=0.10, min_interior_nodes=8,
        interior_max_standoff=1.60,
        interior_loop_clearance=0.25, interior_loop_smoothing=0.05,
        return_metadata=False):
    """Plan a continuous route on a reachable worktop-edge graph.

    A narrow preferred-standoff band supplies the route spine. A wider task-area
    graph supplies an alternate return lane, allowing wall-backed worktops to
    form a closed route without reversing over the same grid edges.
    """
    if not surfaces:
        raise ValueError("worktop edge graph requires at least one surface")
    ctx = _reachable_context(positions)
    surface_groups = _functional_area_groups(
        surfaces, merge_distance=float(surface_merge_distance))
    if not surface_groups:
        surface_groups = [[surface] for surface in surfaces]
    preferred_standoff = float(preferred_standoff)
    tolerance = max(float(standoff_tolerance), ctx["step"] * 0.51)
    minimum = float(min_standoff)
    maximum = float(max_standoff)
    if not minimum < preferred_standoff < maximum:
        raise ValueError("preferred worktop standoff must lie inside the route band")
    interior_max_standoff = float(interior_max_standoff)
    if interior_max_standoff <= 0.0:
        raise ValueError("workspace interior standoff limit must be positive")

    node_distances = {}
    nearest_surfaces = {}
    for key, point in ctx["key_to_point"].items():
        distances = [
            _surface_footprint_distance(point, surface) for surface in surfaces
        ]
        nearest = min(range(len(distances)), key=lambda index: distances[index])
        node_distances[key] = float(distances[nearest])
        nearest_surfaces[key] = int(nearest)
    band_keys_before_interior = {
        key for key, distance in node_distances.items()
        if minimum <= distance <= maximum
    }
    interior_hull = None
    interior_keys = set(ctx["keys"])
    if bool(workspace_interior_only):
        interior_hull = _surface_footprint_hull(surfaces)
        interior_keys = {
            key for key, point in ctx["key_to_point"].items()
            if _inside_surface_hull(
                point, interior_hull, margin=float(interior_hull_margin))
            and node_distances[key] <= interior_max_standoff
        }
        if len(interior_keys) < int(min_interior_nodes):
            raise ValueError(
                "functional workspace interior has only "
                f"{len(interior_keys)} reachable nodes; requires "
                f"{int(min_interior_nodes)}")
        interior_positions = [
            {
                "x": float(ctx["key_to_point"][key][0]),
                "y": 0.0,
                "z": float(ctx["key_to_point"][key][1]),
            }
            for key in sorted(interior_keys)
        ]
        try:
            interior_loop = skeleton_navigation_loop_polyline(
                interior_positions,
                objects=surfaces,
                seed=int(seed),
                clearance=float(interior_loop_clearance),
                smoothing=float(interior_loop_smoothing),
                functional_zone_max_distance=float(extension_max_standoff),
                functional_zone_merge_distance=float(surface_merge_distance),
            )
        except ValueError as exc:
            raise ValueError(
                "functional workspace interior has no one-way loop") from exc
        interior_length = _polyline_length(interior_loop)
        if interior_length < float(min_length):
            raise ValueError(
                f"workspace interior loop is only {interior_length:.2f}m; "
                f"requires {float(min_length):.2f}m")
        outside_count = sum(
            not _inside_surface_hull(
                point, interior_hull, margin=float(interior_hull_margin))
            for point in interior_loop
        )
        if outside_count:
            raise ValueError(
                "workspace interior loop leaves the functional hull at "
                f"{outside_count} points")
        route_standoffs = [
            min(_surface_footprint_distance(point, surface)
                for surface in surfaces)
            for point in interior_loop
        ]
        outside_standoff_count = sum(
            distance > interior_max_standoff + 1e-6
            for distance in route_standoffs
        )
        if outside_standoff_count:
            raise ValueError(
                "workspace interior loop leaves the worktop proximity band at "
                f"{outside_standoff_count} points")
        surface_coverage = sum(
            min(_surface_footprint_distance(point, surface)
                for point in interior_loop) <= float(extension_max_standoff)
            for surface in surfaces
        )
        surface_group_coverage = sum(
            min(_surface_footprint_distance(point, surface)
                for point in interior_loop for surface in group)
            <= float(extension_max_standoff)
            for group in surface_groups
        )
        nearest_surface_count = len({
            min(
                range(len(surfaces)),
                key=lambda index: _surface_footprint_distance(
                    point, surfaces[index]),
            )
            for point in interior_loop
        })
        interior_graph = {
            key: set(_grid_neighbors(key, interior_keys))
            for key in interior_keys
        }
        interior_components = _graph_components(interior_graph)
        interior_tree = cKDTree(np.asarray([
            ctx["key_to_point"][key] for key in interior_keys
        ]))
        interior_deviation = float(np.max(
            interior_tree.query(interior_loop, k=1)[0]))
        metadata = {
            "route_graph_mode": "worktop_edge_graph",
            "route_graph_topology": "interior_loop",
            "route_graph_direction_reversed": bool(
                random.Random(int(seed)).random() < 0.5),
            "route_graph_closed": True,
            "route_graph_grid_step_m": float(ctx["step"]),
            "route_graph_candidate_node_count_before_interior": len(
                band_keys_before_interior),
            "route_graph_candidate_node_count": len(interior_keys),
            "route_graph_component_count": len(interior_components),
            "route_graph_selected_component_nodes": max(
                map(len, interior_components), default=0),
            "route_graph_spine_source": "interior_free_space_loop",
            "route_graph_skeleton_nodes_before_pruning": 0,
            "route_graph_skeleton_node_count": 0,
            "route_graph_endpoint_count": 0,
            "route_graph_junction_count": 0,
            "route_graph_skeleton_has_cycle": True,
            "route_graph_surface_count": len(surfaces),
            "route_graph_surface_covered": surface_coverage,
            "route_graph_nearest_surface_count": nearest_surface_count,
            "route_graph_surface_coverage": (
                surface_coverage / len(surfaces)),
            "route_graph_functional_surface_zone_count": len(surface_groups),
            "route_graph_functional_surface_zone_covered": (
                surface_group_coverage),
            "route_graph_functional_surface_coverage": (
                surface_group_coverage / len(surface_groups)),
            "route_graph_main_length_m": float(interior_length),
            "route_graph_lead_in_length_m": 0.0,
            "route_graph_lead_out_length_m": 0.0,
            "route_graph_return_length_m": 0.0,
            "route_graph_return_separation_m": 0.0,
            "route_graph_standoff_preferred_m": preferred_standoff,
            "route_graph_standoff_mean_m": float(np.mean(route_standoffs)),
            "route_graph_standoff_std_m": float(np.std(route_standoffs)),
            "route_graph_standoff_min_m": float(np.min(route_standoffs)),
            "route_graph_standoff_max_m": float(np.max(route_standoffs)),
            "route_graph_reachable_deviation_max_m": interior_deviation,
            "route_graph_band_deviation_max_m": interior_deviation,
            "route_graph_workspace_interior_only": True,
            "route_graph_interior_hull_margin_m": float(
                interior_hull_margin),
            "route_graph_interior_standoff_limit_m": float(
                interior_max_standoff),
            "route_graph_interior_reachable_node_count": len(interior_keys),
            "route_graph_outside_interior_point_count": 0,
            "route_graph_outside_interior_standoff_point_count": 0,
        }
        if return_metadata:
            return interior_loop, metadata
        return interior_loop
    if bool(workspace_interior_only):
        band_keys = {
            key for key in interior_keys
            if minimum <= node_distances[key] <= float(extension_max_standoff)
        }
    else:
        band_keys = band_keys_before_interior
    band_graph = {
        key: set(_grid_neighbors(key, band_keys)) for key in band_keys
    }
    band_graph = {key: neighbors for key, neighbors in band_graph.items() if neighbors}
    if len(band_graph) < 3:
        raise ValueError("worktop task-area graph is too small")

    candidates = []
    component_count = 0
    for band_component in _graph_components(band_graph):
        component_count += 1
        preferred_keys = {
            key for key in band_component
            if abs(node_distances[key] - preferred_standoff) <= tolerance
        }
        skeleton_sources = [
            ("preferred_band", _skeletonize_grid_keys(preferred_keys)),
            ("task_band", _skeletonize_grid_keys(band_component)),
        ]
        if bool(workspace_interior_only):
            skeleton_sources.append(("interior_band", set(band_component)))
        seen_skeletons = set()
        for spine_source, skeleton_keys in skeleton_sources:
            signature = frozenset(skeleton_keys)
            if not signature or signature in seen_skeletons:
                continue
            seen_skeletons.add(signature)
            skeleton_graph = _band_skeleton_graph(
                skeleton_keys, band_component)
            for skeleton_component in _graph_components(skeleton_graph):
                component_graph = {
                    key: skeleton_graph[key] & skeleton_component
                    for key in skeleton_component
                }
                before_pruning = len(component_graph)
                if spine_source != "interior_band":
                    component_graph = _prune_short_skeleton_branches(
                        component_graph,
                        int(math.ceil(
                            float(max_spur_length) / ctx["step"])),
                    )
                if len(component_graph) < 2:
                    continue
                first = _graph_farthest(component_graph, min(component_graph))
                last = _graph_farthest(component_graph, first)
                main = _weighted_graph_path(
                    component_graph,
                    first,
                    last,
                    node_distances,
                    preferred_standoff,
                    tolerance,
                    turn_cost=turn_cost,
                    standoff_cost=standoff_cost,
                )
                main = _expand_graph_diagonals(
                    main, band_component, node_distances, preferred_standoff)
                main_points = np.asarray(
                    [ctx["key_to_point"][key] for key in main], dtype=float)
                main_length = _polyline_length(main_points)
                if main_length <= ctx["step"]:
                    continue
                surface_coverage = _surface_coverage_for_keys(
                    main, ctx["key_to_point"], surfaces, maximum)
                surface_group_coverage = _surface_group_coverage_for_keys(
                    main, ctx["key_to_point"], surface_groups, maximum)
                standoff_error = float(np.mean([
                    abs(node_distances[key] - preferred_standoff)
                    for key in main
                ]))
                degrees = [
                    len(component_graph[key]) for key in component_graph
                ]
                score = (
                    surface_group_coverage,
                    surface_coverage,
                    main_length,
                    -standoff_error,
                    random.Random(int(seed) + len(candidates)).random(),
                ) if bool(workspace_interior_only) else (
                    surface_group_coverage,
                    surface_coverage,
                    -standoff_error,
                    main_length,
                    random.Random(int(seed) + len(candidates)).random(),
                )
                candidates.append({
                    "score": score,
                    "band_component": band_component,
                    "spine_source": spine_source,
                    "main": main,
                    "main_length": main_length,
                    "surface_coverage": surface_coverage,
                    "surface_group_coverage": surface_group_coverage,
                    "nearest_surface_count": len({
                        nearest_surfaces[key] for key in main
                    }),
                    "standoff_error": standoff_error,
                    "skeleton_nodes_before_pruning": before_pruning,
                    "skeleton_node_count": len(component_graph),
                    "endpoint_count": sum(
                        degree == 1 for degree in degrees),
                    "junction_count": sum(
                        degree >= 3 for degree in degrees),
                    "skeleton_has_cycle": (
                        sum(degrees) // 2 >= len(component_graph)
                    ),
                })
    if not candidates:
        component_sizes = sorted(
            (len(component) for component in _graph_components(band_graph)),
            reverse=True,
        )
        raise ValueError(
            "no continuous preferred-standoff worktop spine: "
            f"band_nodes={len(band_graph)}, "
            f"components={component_sizes[:6]}, "
            f"interior_nodes={len(interior_keys)}")
    selected = max(candidates, key=lambda candidate: candidate["score"])
    main = selected["main"]
    selected_band_graph = {
        key: band_graph[key] & selected["band_component"]
        for key in selected["band_component"]
    }
    return_graph = selected_band_graph
    if bool(workspace_interior_only):
        return_keys = {
            key for key in interior_keys
            if minimum <= node_distances[key] <= float(extension_max_standoff)
        }
        wider_graph = {
            key: set(_grid_neighbors(key, return_keys)) for key in return_keys
        }
        wider_graph = {
            key: neighbors for key, neighbors in wider_graph.items()
            if neighbors
        }
        for component in _graph_components(wider_graph):
            if main[0] in component and main[-1] in component:
                return_graph = {
                    key: wider_graph[key] & component for key in component
                }
                break
    separation_steps = int(round(float(return_separation) / ctx["step"]))
    return_route, used_separation_steps = _alternate_band_return(
        return_graph,
        main,
        node_distances,
        preferred_standoff,
        tolerance,
        separation_steps,
        min_return_length_ratio,
        turn_cost,
        standoff_cost,
    )
    closed = return_route is not None
    if require_closed and not closed:
        raise ValueError("worktop edge graph has no disjoint return lane")
    lead_in = [main[0]]
    lead_out = [main[-1]]
    route_allowed_keys = set(selected["band_component"])
    if closed:
        route_keys = main + return_route[1:]
        route_allowed_keys |= set(return_route)
    else:
        extension_keys = {
            key for key in ctx["keys"]
            if minimum <= node_distances[key] <= float(extension_max_standoff)
        } & interior_keys
        extension_steps = int(math.ceil(float(open_extension) / ctx["step"]))
        lead_in = _open_route_extension(
            main[0],
            main[1],
            extension_keys,
            set(main[1:]),
            node_distances,
            preferred_standoff,
            extension_steps,
            beam_width=extension_beam_width,
        )
        lead_out = _open_route_extension(
            main[-1],
            main[-2],
            extension_keys,
            set(main[:-1]) | set(lead_in[1:]),
            node_distances,
            preferred_standoff,
            extension_steps,
            beam_width=extension_beam_width,
        )
        route_keys = [*reversed(lead_in), *main[1:], *lead_out[1:]]
        route_allowed_keys |= set(lead_in) | set(lead_out)
    reverse_route = False
    if closed:
        reverse_route = random.Random(int(seed)).random() < 0.5
    elif len(lead_in) != len(lead_out):
        # Enter from the constrained end and finish toward open space, leaving
        # forward arc length for the final observation window.
        reverse_route = len(lead_in) > len(lead_out)
    else:
        reverse_route = random.Random(int(seed)).random() < 0.5
    if reverse_route:
        if closed:
            route_keys = [route_keys[0], *reversed(route_keys[1:-1]), route_keys[0]]
        else:
            route_keys = list(reversed(route_keys))
    points = np.asarray(
        [ctx["key_to_point"][key] for key in route_keys], dtype=float)
    if closed:
        rounded = _rounded_closed_polyline(points, radius=rounding_radius)
    else:
        rounded = _rounded_open_polyline(points, radius=rounding_radius)
    reachable_tree = cKDTree(np.asarray(list(ctx["key_to_point"].values())))
    band_tree = cKDTree(np.asarray([
        ctx["key_to_point"][key] for key in route_allowed_keys
    ]))
    reachable_deviation = float(np.max(reachable_tree.query(rounded, k=1)[0]))
    band_deviation = float(np.max(band_tree.query(rounded, k=1)[0]))
    if (reachable_deviation > ctx["step"] * 0.80
            or band_deviation > ctx["step"] * 0.80):
        rounded = points
        reachable_deviation = 0.0
        band_deviation = 0.0
    length = _polyline_length(rounded)
    if length < float(min_length):
        raise ValueError(
            f"worktop edge route is only {length:.2f}m; "
            f"requires {float(min_length):.2f}m")

    route_standoffs = [
        min(_surface_footprint_distance(point, surface) for surface in surfaces)
        for point in rounded
    ]
    outside_interior_count = (
        sum(not _inside_surface_hull(
            point, interior_hull, margin=float(interior_hull_margin))
            for point in rounded)
        if interior_hull is not None else 0
    )
    if outside_interior_count:
        raise ValueError(
            "rounded route leaves the functional workspace interior at "
            f"{outside_interior_count} points")
    metadata = {
        "route_graph_mode": "worktop_edge_graph",
        "route_graph_topology": "closed_return" if closed else "open",
        "route_graph_direction_reversed": bool(reverse_route),
        "route_graph_closed": bool(closed),
        "route_graph_grid_step_m": float(ctx["step"]),
        "route_graph_candidate_node_count_before_interior": len(
            band_keys_before_interior),
        "route_graph_candidate_node_count": len(band_graph),
        "route_graph_workspace_interior_only": bool(workspace_interior_only),
        "route_graph_interior_hull_margin_m": float(interior_hull_margin),
        "route_graph_interior_reachable_node_count": len(interior_keys),
        "route_graph_outside_interior_point_count": outside_interior_count,
        "route_graph_component_count": component_count,
        "route_graph_selected_component_nodes": len(selected["band_component"]),
        "route_graph_spine_source": selected["spine_source"],
        "route_graph_skeleton_nodes_before_pruning": selected[
            "skeleton_nodes_before_pruning"],
        "route_graph_skeleton_node_count": selected["skeleton_node_count"],
        "route_graph_endpoint_count": selected["endpoint_count"],
        "route_graph_junction_count": selected["junction_count"],
        "route_graph_skeleton_has_cycle": selected["skeleton_has_cycle"],
        "route_graph_surface_count": len(surfaces),
        "route_graph_surface_covered": selected["surface_coverage"],
        "route_graph_nearest_surface_count": selected[
            "nearest_surface_count"],
        "route_graph_surface_coverage": (
            selected["surface_coverage"] / len(surfaces)),
        "route_graph_functional_surface_zone_count": len(surface_groups),
        "route_graph_functional_surface_zone_covered": selected[
            "surface_group_coverage"],
        "route_graph_functional_surface_coverage": (
            selected["surface_group_coverage"] / len(surface_groups)),
        "route_graph_main_length_m": float(selected["main_length"]),
        "route_graph_lead_in_length_m": float(
            (len(lead_in) - 1) * ctx["step"]),
        "route_graph_lead_out_length_m": float(
            (len(lead_out) - 1) * ctx["step"]),
        "route_graph_return_length_m": float(
            _polyline_length(np.asarray([
                ctx["key_to_point"][key] for key in return_route
            ], dtype=float)) if return_route else 0.0),
        "route_graph_return_separation_m": float(
            (used_separation_steps or 0) * ctx["step"]),
        "route_graph_standoff_preferred_m": preferred_standoff,
        "route_graph_standoff_mean_m": float(np.mean(route_standoffs)),
        "route_graph_standoff_std_m": float(np.std(route_standoffs)),
        "route_graph_standoff_min_m": float(np.min(route_standoffs)),
        "route_graph_standoff_max_m": float(np.max(route_standoffs)),
        "route_graph_reachable_deviation_max_m": reachable_deviation,
        "route_graph_band_deviation_max_m": band_deviation,
    }
    if return_metadata:
        return rounded, metadata
    return rounded


def worktop_corridor_polyline(positions, surfaces, seed=0,
                              min_standoff=0.40, max_standoff=1.60,
                              min_length=2.0):
    """Return a long, non-backtracking reachable path beside work surfaces."""
    if not surfaces:
        raise ValueError("worktop corridor requires at least one surface")
    ctx = _reachable_context(positions)
    candidate_keys = {
        key for key, point in ctx["key_to_point"].items()
        if min(_surface_footprint_distance(point, surface)
               for surface in surfaces)
        <= float(max_standoff)
        and min(_surface_footprint_distance(point, surface)
                for surface in surfaces)
        >= float(min_standoff)
    }
    graph = {
        key: set(_grid_neighbors(key, candidate_keys))
        for key in candidate_keys
    }
    graph = {key: neighbors for key, neighbors in graph.items() if neighbors}
    if len(graph) < 2:
        raise ValueError("worktop-adjacent reachable corridor is too small")

    components = []
    remaining = set(graph)
    while remaining:
        root = min(remaining)
        component = {root}
        frontier = [root]
        while frontier:
            node = frontier.pop()
            for neighbor in graph[node]:
                if neighbor not in component:
                    component.add(neighbor)
                    frontier.append(neighbor)
        remaining -= component
        components.append(component)

    routes = []
    for component in components:
        component_graph = {
            key: graph[key] & component for key in component
        }
        first = _graph_farthest(component_graph, min(component))
        last = _graph_farthest(component_graph, first)
        route = _graph_shortest_path(component_graph, first, last)
        points = np.array(
            [ctx["key_to_point"][key] for key in route], dtype=float)
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        routes.append((length, points))
    length, route = max(routes, key=lambda item: item[0])
    if length < float(min_length):
        raise ValueError(
            f"worktop corridor is only {length:.2f}m; "
            f"requires {float(min_length):.2f}m")
    if random.Random(int(seed)).random() < 0.5:
        route = route[::-1]
    return route


def _rounded_open_polyline(points, radius=0.12):
    """Simplify and round an open route while preserving its endpoints."""
    deduped = []
    for point in points:
        point = np.asarray(point, dtype=float)
        if (not deduped
                or float(np.linalg.norm(point - deduped[-1])) > 1e-6):
            deduped.append(point)
    points = np.asarray(deduped, dtype=float)
    if len(points) < 3:
        return points

    simplified = [points[0]]
    for index in range(1, len(points) - 1):
        incoming = points[index] - simplified[-1]
        outgoing = points[index + 1] - points[index]
        cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        dot = float(np.dot(incoming, outgoing))
        if abs(cross) <= 1e-9 and dot > 0:
            continue
        simplified.append(points[index])
    simplified.append(points[-1])
    points = np.asarray(simplified, dtype=float)

    rounded = [points[0]]
    for index in range(1, len(points) - 1):
        previous, corner, following = points[index - 1:index + 2]
        incoming = previous - corner
        outgoing = following - corner
        incoming_length = float(np.linalg.norm(incoming))
        outgoing_length = float(np.linalg.norm(outgoing))
        if incoming_length <= 1e-6 or outgoing_length <= 1e-6:
            rounded.append(corner)
            continue
        incoming /= incoming_length
        outgoing /= outgoing_length
        corner_radius = min(
            float(radius), incoming_length * 0.40, outgoing_length * 0.40)
        entry = corner + incoming * corner_radius
        exit_point = corner + outgoing * corner_radius
        rounded.append(entry)
        for sample in range(1, 5):
            progress = sample / 5.0
            inverse = 1.0 - progress
            rounded.append(
                inverse * inverse * entry
                + 2.0 * inverse * progress * corner
                + progress * progress * exit_point
            )
        rounded.append(exit_point)
    rounded.append(points[-1])
    return np.asarray(rounded, dtype=float)


def corridor_route_polyline(corridor, start, goal):
    """Route between two poses along a precomputed worktop corridor."""
    corridor = np.asarray(corridor, dtype=float)
    start = np.asarray(start[:2], dtype=float)
    goal = np.asarray(goal[:2], dtype=float)
    if len(corridor) < 2:
        return _rounded_open_polyline([start, goal])

    cumulative = np.concatenate((
        np.array([0.0]),
        np.cumsum(np.linalg.norm(np.diff(corridor, axis=0), axis=1)),
    ))

    def project(point):
        best = None
        for index, (left, right) in enumerate(zip(corridor, corridor[1:])):
            delta = right - left
            length_sq = float(np.dot(delta, delta))
            progress = 0.0 if length_sq <= 1e-12 else float(np.clip(
                np.dot(point - left, delta) / length_sq, 0.0, 1.0))
            projected = left + delta * progress
            distance_sq = float(np.dot(point - projected, point - projected))
            along = cumulative[index] + math.sqrt(length_sq) * progress
            candidate = (distance_sq, along, index, progress, projected)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        return best

    start_projection = project(start)
    goal_projection = project(goal)
    _, start_along, start_index, _, start_point = start_projection
    _, goal_along, goal_index, _, goal_point = goal_projection

    if start_along <= goal_along:
        route = [start, start_point]
        route.extend(corridor[start_index + 1:goal_index + 1])
        route.extend((goal_point, goal))
    else:
        route = [start, start_point]
        route.extend(corridor[goal_index + 1:start_index + 1][::-1])
        route.extend((goal_point, goal))
    return _rounded_open_polyline(route)


def reachable_shortest_polyline(positions, start, goal):
    """Connect arbitrary XZ endpoints through the reachable navigation grid."""
    ctx = _reachable_context(positions)
    keys = set(ctx["key_to_point"])
    graph = {key: set(_grid_neighbors(key, keys)) for key in keys}
    ordered_keys = sorted(keys)
    points = np.array([ctx["key_to_point"][key] for key in ordered_keys])
    tree = cKDTree(points)

    def xz(value):
        if isinstance(value, dict):
            return np.array([float(value["x"]), float(value["z"])])
        return np.asarray(value[:2], dtype=float)

    start_point = xz(start)
    goal_point = xz(goal)
    start_key = ordered_keys[int(tree.query(start_point, k=1)[1])]
    goal_key = ordered_keys[int(tree.query(goal_point, k=1)[1])]
    route = _graph_shortest_path(graph, start_key, goal_key)
    polyline = [start_point]
    polyline.extend(ctx["key_to_point"][key] for key in route)
    polyline.append(goal_point)
    return _rounded_open_polyline(polyline)


def _perimeter_anchors(key_to_point, anchor_count: int):
    keys = set(key_to_point)
    boundary = [
        k for k in keys
        if sum(1 for _ in _grid_neighbors(k, keys)) < 4
    ] or list(keys)
    pts = np.array(list(key_to_point.values()))
    center = pts.mean(axis=0)

    bins = {}
    for key in boundary:
        p = key_to_point[key]
        angle = (math.atan2(p[1] - center[1], p[0] - center[0]) + 2 * math.pi) % (2 * math.pi)
        bi = int(angle / (2 * math.pi) * anchor_count)
        radius = float(np.linalg.norm(p - center))
        if bi not in bins or radius > bins[bi][0]:
            bins[bi] = (radius, key)

    anchors = [bins[i][1] for i in sorted(bins)]
    deduped = []
    for key in anchors:
        if not deduped or deduped[-1] != key:
            deduped.append(key)
    if len(deduped) > 1 and deduped[0] == deduped[-1]:
        deduped.pop()
    return deduped


def _shortest_grid_path(start, goal, key_to_point, center, max_radius):
    keys = set(key_to_point)
    frontier = [(0.0, 0.0, start)]
    came_from = {start: None}
    best = {start: 0.0}

    def heuristic(key):
        p = key_to_point[key]
        g = key_to_point[goal]
        return abs(p[0] - g[0]) + abs(p[1] - g[1])

    while frontier:
        _, cost, key = heapq.heappop(frontier)
        if key == goal:
            break
        if cost > best.get(key, float("inf")):
            continue
        for nb in _grid_neighbors(key, keys):
            radius = float(np.linalg.norm(key_to_point[nb] - center))
            interior_penalty = (max_radius - radius) / max(max_radius, 1e-6)
            new_cost = cost + 1.0 + interior_penalty * 0.35
            if new_cost < best.get(nb, float("inf")):
                best[nb] = new_cost
                came_from[nb] = key
                heapq.heappush(frontier, (new_cost + heuristic(nb), new_cost, nb))

    if goal not in came_from:
        raise ValueError(f"no reachable-grid path between {start} and {goal}")

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    return list(reversed(path))


def _reachable_perimeter_polyline(positions, anchor_count: int = 24):
    _, key_to_point = _reachable_grid(positions)
    anchors = _perimeter_anchors(key_to_point, anchor_count)
    if len(anchors) < 3:
        raise ValueError("not enough perimeter anchors to build a walking loop")

    pts = np.array(list(key_to_point.values()))
    center = pts.mean(axis=0)
    max_radius = max(float(np.linalg.norm(p - center)) for p in pts)

    route = []
    for i, start in enumerate(anchors):
        goal = anchors[(i + 1) % len(anchors)]
        segment = _shortest_grid_path(start, goal, key_to_point, center, max_radius)
        if route:
            segment = segment[1:]
        route.extend(segment)
    if route and route[0] != route[-1]:
        segment = _shortest_grid_path(route[-1], route[0], key_to_point, center, max_radius)
        route.extend(segment[1:])

    poly = [key_to_point[k] for k in route]
    deduped = []
    for p in poly:
        if not deduped or float(np.linalg.norm(p - deduped[-1])) > 1e-6:
            deduped.append(p)
    return np.array(deduped)


def _polyline_cumulative(poly):
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-6])
    poly = poly[keep]
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg)])
    return poly, cum_len


def _interp_polyline(poly, cum_len, distance):
    total = float(cum_len[-1])
    d = min(max(distance, 0.0), total)
    x = np.interp(d, cum_len, poly[:, 0])
    z = np.interp(d, cum_len, poly[:, 1])
    return np.array([x, z]), total


def _interp_cyclic_polyline(poly, cum_len, distance):
    total = float(cum_len[-1])
    if total <= 0.0:
        return np.asarray(poly[0], dtype=float), total
    d = float(distance) % total
    x = np.interp(d, cum_len, poly[:, 0])
    z = np.interp(d, cum_len, poly[:, 1])
    return np.array([x, z]), total


def _poses_from_closed_polyline(poly, fps, speed, lookahead,
                                max_yaw_rate=60.0,
                                turn_smoothing_distance=0.35):
    """Sample a closed path with body heading and curvature-aware speed.

    Straight segments retain the requested walking speed. Tight turns are
    traversed more slowly so the body's angular velocity remains human-scale,
    analogous to Qwen-RobotNav's variable substeps and smooth deceleration.
    """
    poly = np.asarray(poly, dtype=float)
    if float(np.linalg.norm(poly[0] - poly[-1])) > 1e-6:
        poly = np.vstack([poly, poly[0]])
    poly, cum_len = _polyline_cumulative(poly)
    total = float(cum_len[-1])
    fps = max(float(fps), 1.0)
    nominal_speed = max(float(speed), 1e-3)
    dense_step = min(0.01, nominal_speed / fps / 3.0)
    dense_count = max(256, int(math.ceil(total / dense_step)))
    dense_distances = np.linspace(0.0, total, dense_count, endpoint=False)
    actual_dense_step = total / dense_count
    tangent_window = max(min(float(lookahead), 0.15), actual_dense_step * 4.0)

    dense_yaws = []
    for distance in dense_distances:
        before, _ = _interp_cyclic_polyline(
            poly, cum_len, distance - tangent_window / 2.0)
        after, _ = _interp_cyclic_polyline(
            poly, cum_len, distance + tangent_window / 2.0)
        tangent = after - before
        dense_yaws.append(heading_to(float(tangent[0]), float(tangent[1])))
    unwrapped_yaws = np.asarray(_unwrap_yaws(dense_yaws), dtype=float)
    closing_yaw = unwrapped_yaws[-1] + shortest_delta(
        unwrapped_yaws[-1] % 360.0, dense_yaws[0])
    yaw_steps = np.diff(np.append(unwrapped_yaws, closing_yaw))
    curvature = np.maximum(
        np.abs(yaw_steps), np.abs(np.roll(yaw_steps, 1))) / actual_dense_step

    angular_limit = max(float(max_yaw_rate), 1.0)
    speed_limit = np.minimum(
        nominal_speed,
        angular_limit * 0.85 / np.maximum(curvature, 1e-6),
    )
    smoothing_samples = max(
        1, int(round(float(turn_smoothing_distance) / actual_dense_step)))
    speed_limit = ndimage.minimum_filter1d(
        speed_limit, size=smoothing_samples * 2 + 1, mode="wrap")
    speed_limit = ndimage.gaussian_filter1d(
        speed_limit, sigma=max(smoothing_samples / 2.0, 1.0), mode="wrap")
    speed_limit = np.maximum(speed_limit, 0.03)

    segment_times = actual_dense_step / speed_limit
    cumulative_time = np.concatenate([[0.0], np.cumsum(segment_times)])
    total_time = float(cumulative_time[-1])
    frame_count = max(2, int(math.ceil(total_time * fps)))
    frame_times = np.linspace(0.0, total_time, frame_count, endpoint=False)
    distance_axis = np.append(dense_distances, total)
    distances = np.interp(frame_times, cumulative_time, distance_axis)
    yaw_axis = np.append(unwrapped_yaws, closing_yaw)
    sampled_yaws = np.interp(distances, distance_axis, yaw_axis)

    poses = []
    for distance, yaw in zip(distances, sampled_yaws):
        point, _ = _interp_cyclic_polyline(poly, cum_len, distance)
        poses.append((float(point[0]), float(point[1]), float(yaw) % 360.0))
    return poses


def trajectory_quality(poses, fps, high_frequency_hz=3.0):
    """Return kinematic diagnostics for one cyclic pose sequence."""
    poses = list(poses)
    if len(poses) < 3:
        raise ValueError("trajectory quality requires at least 3 poses")
    fps = max(float(fps), 1.0)
    positions = np.array([[x, z] for x, z, _ in poses], dtype=float)
    next_positions = np.roll(positions, -1, axis=0)
    motion = next_positions - positions
    steps = np.linalg.norm(motion, axis=1)
    moving = steps > 1e-6
    motion_yaws = np.array([
        heading_to(float(delta[0]), float(delta[1])) if is_moving else poses[i][2]
        for i, (delta, is_moving) in enumerate(zip(motion, moving))
    ])
    offsets = np.array([
        abs(shortest_delta(float(yaw), float(motion_yaw)))
        for (_, _, yaw), motion_yaw in zip(poses, motion_yaws)
    ])

    yaws = np.asarray(_unwrap_yaws([yaw for _, _, yaw in poses]), dtype=float)
    closing_yaw = yaws[-1] + shortest_delta(yaws[-1] % 360.0, poses[0][2])
    yaw_rate = np.diff(np.append(yaws, closing_yaw)) * fps
    yaw_acceleration = np.diff(np.append(yaw_rate, yaw_rate[0])) * fps
    active_rate = yaw_rate[np.abs(yaw_rate) >= 1.0]
    sign_flips = int(np.sum(active_rate[1:] * active_rate[:-1] < 0.0))

    centered_rate = yaw_rate - float(np.mean(yaw_rate))
    spectrum = np.abs(np.fft.rfft(centered_rate)) ** 2
    frequencies = np.fft.rfftfreq(len(centered_rate), d=1.0 / fps)
    total_power = float(np.sum(spectrum[1:]))
    high_power = float(np.sum(spectrum[frequencies >= float(high_frequency_hz)]))
    high_frequency_ratio = high_power / total_power if total_power > 1e-9 else 0.0

    moving_offsets = offsets[moving]
    return {
        "distance_m": float(np.sum(steps)),
        "stationary_ratio": float(np.mean(~moving)),
        "backward_ratio": float(np.mean(moving_offsets > 90.0)) if len(moving_offsets) else 0.0,
        "view_offset_p95_deg": float(np.percentile(moving_offsets, 95)) if len(moving_offsets) else 0.0,
        "view_offset_max_deg": float(np.max(moving_offsets)) if len(moving_offsets) else 0.0,
        "yaw_rate_p95_deg_s": float(np.percentile(np.abs(yaw_rate), 95)),
        "yaw_rate_max_deg_s": float(np.max(np.abs(yaw_rate))),
        "yaw_acceleration_max_deg_s2": float(np.max(np.abs(yaw_acceleration))),
        "yaw_rate_sign_flips": sign_flips,
        "high_frequency_yaw_ratio": high_frequency_ratio,
    }


def validate_trajectory_quality(poses, fps, thresholds=None):
    """Reject trajectories that would render as backward or jittery motion."""
    limits = {
        "backward_ratio": 0.0,
        "stationary_ratio": 0.01,
        "view_offset_p95_deg": 15.0,
        "yaw_rate_p95_deg_s": 60.0,
        "yaw_rate_max_deg_s": 90.0,
        "yaw_acceleration_max_deg_s2": 300.0,
        "high_frequency_yaw_ratio": 0.05,
    }
    limits.update(thresholds or {})
    metrics = trajectory_quality(poses, fps)
    failures = [
        f"{name}={metrics[name]:.4g}>{limit:.4g}"
        for name, limit in limits.items()
        if metrics[name] > float(limit) + 1e-9
    ]
    if failures:
        raise ValueError("unnatural trajectory: " + ", ".join(failures))
    return metrics


def _position_xz(item):
    pos = item.get("position", item)
    return np.array([float(pos["x"]), float(pos["z"])])


def _look_candidates(objects, object_types=None):
    if object_types:
        weights = {
            object_type: DEFAULT_LOOK_OBJECT_WEIGHTS.get(object_type, 1.0)
            for object_type in object_types
        }
    else:
        weights = DEFAULT_LOOK_OBJECT_WEIGHTS
    candidates = []
    for obj in objects or []:
        object_type = obj.get("objectType")
        if object_type not in weights or not obj.get("position"):
            continue
        candidates.append({
            "xz": _position_xz(obj),
            "weight": float(weights[object_type]),
        })
    return candidates


def _natural_view_yaw(position_xz, tangent_yaw, look_candidates=None,
                      center_xz=None, max_turn_degrees=90.0,
                      center_max_turn_degrees=None,
                      min_distance=0.55, max_distance=4.0,
                      object_strength=0.8, center_strength=0.3):
    """Blend path heading toward semantically plausible human-looking targets."""
    best = None
    turn_limit = abs(float(max_turn_degrees))
    for candidate in look_candidates or []:
        vec = candidate["xz"] - position_xz
        dist = float(np.linalg.norm(vec))
        if dist < min_distance or dist > max_distance:
            continue
        target_yaw = heading_to(float(vec[0]), float(vec[1]))
        delta = shortest_delta(tangent_yaw, target_yaw)
        abs_delta = abs(delta)
        if abs_delta > turn_limit:
            continue
        angle_score = 1.0 - abs_delta / max(turn_limit, 1e-6)
        preferred_distance = 1.8
        distance_score = max(
            0.0,
            1.0 - abs(dist - preferred_distance) / max(max_distance - min_distance, 1e-6),
        )
        score = candidate["weight"] * (0.25 + 0.75 * angle_score) * (0.55 + 0.45 * distance_score)
        if best is None or score > best[0]:
            best = (score, delta)

    if best is not None:
        return (tangent_yaw + best[1] * float(object_strength)) % 360.0

    if center_xz is not None:
        vec = center_xz - position_xz
        if float(np.linalg.norm(vec)) > 1e-6:
            center_yaw = heading_to(float(vec[0]), float(vec[1]))
            delta = shortest_delta(tangent_yaw, center_yaw)
            center_turn_limit = (
                turn_limit if center_max_turn_degrees is None
                else abs(float(center_max_turn_degrees))
            )
            if abs(delta) <= center_turn_limit:
                return (tangent_yaw + delta * float(center_strength)) % 360.0

    return tangent_yaw


def _candidate_delta_score(position_xz, tangent_yaw, candidate,
                           max_turn_degrees, min_distance, max_distance):
    vec = candidate["xz"] - position_xz
    dist = float(np.linalg.norm(vec))
    if dist < min_distance or dist > max_distance:
        return None
    target_yaw = heading_to(float(vec[0]), float(vec[1]))
    delta = shortest_delta(tangent_yaw, target_yaw)
    abs_delta = abs(delta)
    turn_limit = abs(float(max_turn_degrees))
    if abs_delta > turn_limit:
        return None
    angle_score = 1.0 - abs_delta / max(turn_limit, 1e-6)
    preferred_distance = 1.8
    distance_score = max(
        0.0,
        1.0 - abs(dist - preferred_distance) / max(max_distance - min_distance, 1e-6),
    )
    score = candidate["weight"] * (0.25 + 0.75 * angle_score) * (0.55 + 0.45 * distance_score)
    return delta, score


def _unwrap_yaws(yaws):
    if not yaws:
        return []
    unwrapped = [float(yaws[0])]
    for yaw in yaws[1:]:
        unwrapped.append(unwrapped[-1] + shortest_delta(unwrapped[-1] % 360.0, yaw))
    return unwrapped


def _attenuate_small_delta(delta, deadband):
    """Continuously damp tiny yaw deltas without freezing and releasing."""
    deadband = abs(float(deadband))
    if deadband <= 0.0:
        return delta
    magnitude = abs(float(delta))
    if magnitude <= 1e-9:
        return 0.0
    return delta * (magnitude / (magnitude + deadband))


def _kalman_smooth_yaws(yaws, fps, process_noise=900.0,
                        measurement_noise=64.0,
                        initial_uncertainty=100.0,
                        deadband_degrees=0.0):
    """Constant-velocity Kalman filter over unwrapped yaw angles."""
    if len(yaws) < 2:
        return yaws

    measurements = _unwrap_yaws(yaws)
    dt = 1.0 / max(float(fps), 1.0)
    f = np.array([[1.0, dt], [0.0, 1.0]])
    h = np.array([[1.0, 0.0]])
    q = float(process_noise) * np.array(
        [[dt ** 4 / 4.0, dt ** 3 / 2.0], [dt ** 3 / 2.0, dt ** 2]]
    )
    r = np.array([[max(float(measurement_noise), 1e-6)]])
    x = np.array([[measurements[0]], [0.0]])
    p = np.array([[float(initial_uncertainty), 0.0],
                  [0.0, float(initial_uncertainty)]])
    eye = np.eye(2)
    deadband = abs(float(deadband_degrees))
    result = []

    for z in measurements:
        x = f @ x
        p = f @ p @ f.T + q
        predicted = float(x[0, 0])
        measurement = predicted + _attenuate_small_delta(float(z) - predicted, deadband)
        innovation = np.array([[measurement]]) - h @ x
        s = h @ p @ h.T + r
        k = p @ h.T @ np.linalg.inv(s)
        x = x + k @ innovation
        p = (eye - k @ h) @ p
        result.append(float(x[0, 0]) % 360.0)
    return result


def _stabilize_view_yaw(poses, fps, max_yaw_rate=35.0,
                        smoothing_frames=18, deadband_degrees=0.0,
                        filter_mode="lowpass",
                        kalman_process_noise=900.0,
                        kalman_measurement_noise=64.0,
                        kalman_initial_uncertainty=100.0):
    """Add head-motion inertia after semantic view-target selection."""
    poses = list(poses)
    if len(poses) < 2:
        return poses

    max_step = abs(float(max_yaw_rate)) / max(float(fps), 1.0)
    deadband = abs(float(deadband_degrees))
    mode = str(filter_mode or "lowpass").lower()
    if mode == "kalman":
        filtered_yaws = _kalman_smooth_yaws(
            [yaw for _, _, yaw in poses],
            fps,
            process_noise=kalman_process_noise,
            measurement_noise=kalman_measurement_noise,
            initial_uncertainty=kalman_initial_uncertainty,
            deadband_degrees=deadband,
        )
        desired = [
            (x, z, yaw)
            for (x, z, _), yaw in zip(poses, filtered_yaws)
        ]
    else:
        alpha = 1.0 / max(float(smoothing_frames), 1.0)
        desired = []
        prev_yaw = poses[0][2]
        for x, z, desired_yaw in poses:
            delta = shortest_delta(prev_yaw, desired_yaw)
            yaw = (prev_yaw + _attenuate_small_delta(delta, deadband) * alpha) % 360.0
            desired.append((x, z, yaw))
            prev_yaw = yaw

    smoothed = []
    prev_yaw = desired[0][2]
    for x, z, desired_yaw in desired:
        delta = shortest_delta(prev_yaw, desired_yaw)
        step = _attenuate_small_delta(delta, deadband)
        if abs(step) > max_step:
            step = max_step * (1 if step > 0 else -1)
        yaw = (prev_yaw + step) % 360.0
        smoothed.append((x, z, yaw))
        prev_yaw = yaw
    return smoothed


def _natural_point_metrics(ctx, point):
    key = _point_key(point, ctx["step"])
    return (
        ctx["neighbor_count"].get(key, 0),
        ctx["clearance"].get(key, 0.0),
    )


def _nearest_reachable_standoff(positions, target_xz, preferred=0.9,
                                min_distance=0.45, ctx=None,
                                min_clearance=0.28,
                                min_open_neighbors=3):
    """Pick a reachable point near, but not on top of, a target."""
    ctx = ctx or _reachable_context(positions)
    scored = []
    for p in positions:
        pxz = _position_xz(p)
        dist = float(np.linalg.norm(pxz - target_xz))
        if dist < min_distance:
            continue
        open_neighbors, clearance = _natural_point_metrics(ctx, p)
        wall_penalty = max(0.0, min_clearance - clearance) * 2.0
        dead_end_penalty = max(0, min_open_neighbors - open_neighbors) * 0.35
        score = abs(dist - preferred) + wall_penalty + dead_end_penalty
        scored.append((score, clearance, open_neighbors, p))
    if scored:
        natural = [
            item for item in scored
            if item[1] >= min_clearance and item[2] >= min_open_neighbors
        ]
        return min(natural or scored, key=lambda item: item[0])[3]
    return min(positions, key=lambda p: float(np.linalg.norm(_position_xz(p) - target_xz)))


def _interior_radial_anchors(positions, ctx, anchor_count):
    """Distributed fallback anchors that avoid perimeter corners."""
    center = ctx["center"]
    candidates = []
    for p in positions:
        pxz = _position_xz(p)
        radius = float(np.linalg.norm(pxz - center))
        open_neighbors, clearance = _natural_point_metrics(ctx, p)
        if open_neighbors < 3:
            continue
        candidates.append((p, pxz, radius, clearance, open_neighbors))
    if not candidates:
        candidates = [
            (p, _position_xz(p), float(np.linalg.norm(_position_xz(p) - center)), 0.0, 0)
            for p in positions
        ]

    radii = [item[2] for item in candidates]
    ideal_radius = float(np.percentile(radii, 70)) if radii else 0.0
    bins = {}
    for p, pxz, radius, clearance, open_neighbors in candidates:
        angle = (math.atan2(pxz[1] - center[1], pxz[0] - center[0]) + 2 * math.pi) % (2 * math.pi)
        bi = int(angle / (2 * math.pi) * anchor_count)
        score = -abs(radius - ideal_radius) + clearance * 0.6 + open_neighbors * 0.04
        if bi not in bins or score > bins[bi][0]:
            bins[bi] = (score, p)

    anchors = [bins[i][1] for i in sorted(bins)]
    deduped = []
    for p in anchors:
        if not deduped or float(np.linalg.norm(_position_xz(p) - _position_xz(deduped[-1]))) > 1e-6:
            deduped.append(p)
    return deduped


def select_tour_waypoints(positions, objects, object_types=None,
                          max_waypoints=6, min_separation=0.75,
                          min_clearance=0.35,
                          min_open_neighbors=3,
                          ctx=None):
    """Choose human-plausible room-tour waypoints near functional scene objects."""
    object_types = tuple(object_types or DEFAULT_TOUR_OBJECT_TYPES)
    ctx = ctx or _reachable_context(positions)
    center = ctx["center"]
    chosen = []

    for object_type in object_types:
        candidates = [
            o for o in objects
            if o.get("objectType") == object_type and o.get("position")
        ]
        if not candidates:
            continue
        # Prefer objects distributed away from the room center: they make better
        # observation stops than tiny central clutter.
        candidates.sort(key=lambda o: float(np.linalg.norm(_position_xz(o) - center)),
                        reverse=True)
        for obj in candidates[:3]:
            wp = _nearest_reachable_standoff(
                positions,
                _position_xz(obj),
                ctx=ctx,
                min_clearance=min_clearance,
                min_open_neighbors=min_open_neighbors,
            )
            wp_xz = _position_xz(wp)
            if any(float(np.linalg.norm(wp_xz - _position_xz(old))) < min_separation
                   for old in chosen):
                continue
            chosen.append(wp)
            break
        if len(chosen) >= max_waypoints:
            break

    if len(chosen) < 4:
        for wp in _interior_radial_anchors(positions, ctx, 12):
            if any(float(np.linalg.norm(_position_xz(wp) - _position_xz(old))) < min_separation
                   for old in chosen):
                continue
            chosen.append(wp)
            if len(chosen) >= max_waypoints:
                break

    def angle(wp):
        pxz = _position_xz(wp)
        return math.atan2(pxz[1] - center[1], pxz[0] - center[0])

    chosen.sort(key=angle)
    return chosen


def _shortest_path_corners(controller, start, goal, allowed_error=0.25):
    event = controller.step(
        action="GetShortestPathToPoint",
        position={"x": start["x"], "y": start["y"], "z": start["z"]},
        target={"x": goal["x"], "y": goal["y"], "z": goal["z"]},
        allowedError=allowed_error,
    )
    if not event.metadata["lastActionSuccess"]:
        raise ValueError(
            "GetShortestPathToPoint failed from "
            f"{start} to {goal}: {event.metadata.get('errorMessage')}"
        )
    return event.metadata["actionReturn"]["corners"]


def _dedupe_positions(points):
    deduped = []
    for p in points:
        item = {"x": float(p["x"]), "y": float(p["y"]), "z": float(p["z"])}
        if not deduped or float(np.linalg.norm(_position_xz(item) - _position_xz(deduped[-1]))) > 1e-5:
            deduped.append(item)
    return deduped


def _poses_from_polyline_points(points, fps, speed, lookahead,
                                look_candidates=None, center_xz=None,
                                look_max_turn_degrees=125.0,
                                look_center_max_turn_degrees=180.0,
                                look_min_distance=0.55,
                                look_max_distance=5.0,
                                look_object_strength=0.9,
                                look_center_strength=0.65,
                                look_max_yaw_rate=40.0,
                                look_smoothing_frames=4,
                                look_deadband_degrees=0.0,
                                look_target_hold_frames=60,
                                look_release_grace_frames=8,
                                look_switch_margin=0.15,
                                look_filter="lowpass",
                                look_kalman_process_noise=900.0,
                                look_kalman_measurement_noise=64.0,
                                look_kalman_initial_uncertainty=100.0):
    poly = np.array([[float(p["x"]), float(p["z"])] for p in points])
    poly, cum_len = _polyline_cumulative(poly)
    step = max(speed / fps, 1e-4)
    total = float(cum_len[-1])
    n = max(2, int(round(total / step)))
    targets = np.linspace(0.0, total, n, endpoint=False)

    poses = []
    yaw_lookahead = max(lookahead, step * 4)
    # Look-target state machine. The camera commits to ONE semantic target and
    # only switches when a rival is clearly better (hysteresis) AND the current
    # target has been held long enough (min-hold). When the held target briefly
    # scores None (walked too close / turned past it), we keep facing the last
    # good yaw for a short grace window instead of snapping to center and back --
    # that "fling out then fling back" was the visible occasional wobble.
    held_candidate = None
    held_since = 0
    grace_left = 0
    last_yaw = None
    min_hold = int(max(1, look_target_hold_frames))
    grace_frames = int(max(0, look_release_grace_frames))
    switch_margin = float(look_switch_margin)
    for frame_i, d in enumerate(targets):
        p, _ = _interp_polyline(poly, cum_len, d)
        nxt, _ = _interp_polyline(poly, cum_len, min(d + yaw_lookahead, total))
        if float(np.linalg.norm(nxt - p)) < 1e-6:
            nxt, _ = _interp_polyline(poly, cum_len, max(d - yaw_lookahead, 0.0))
            tangent = p - nxt
        else:
            tangent = nxt - p
        tangent_yaw = heading_to(float(tangent[0]), float(tangent[1]))

        # Score every visible candidate this frame.
        best = None
        held_scored = None
        for candidate in look_candidates or []:
            scored = _candidate_delta_score(
                p, tangent_yaw, candidate, look_max_turn_degrees,
                look_min_distance, look_max_distance)
            if scored is None:
                continue
            delta, score = scored
            if candidate is held_candidate:
                held_scored = (delta, score)
            if best is None or score > best[0]:
                best = (score, delta, candidate)

        yaw = None
        if held_candidate is not None and held_scored is not None:
            # Keep the held target unless a rival beats it by a clear margin and
            # the min-hold has elapsed. Otherwise stay committed (no argmax flip).
            can_switch = (
                best is not None
                and best[2] is not held_candidate
                and frame_i - held_since >= min_hold
                and best[0] > held_scored[1] * (1.0 + switch_margin)
            )
            if can_switch:
                yaw = (tangent_yaw + best[1] * look_object_strength) % 360.0
                held_candidate = best[2]
                held_since = frame_i
            else:
                yaw = (tangent_yaw + held_scored[0] * look_object_strength) % 360.0
            grace_left = grace_frames
        elif held_candidate is not None and grace_left > 0 and last_yaw is not None:
            # Held target momentarily unavailable: coast on the last good yaw for
            # a few frames rather than snapping to center (kills the round-trip).
            grace_left -= 1
            yaw = last_yaw
        else:
            # No commitment (or grace expired): acquire the best target, else
            # fall back to the gentle path/center heading.
            if best is not None:
                yaw = (tangent_yaw + best[1] * look_object_strength) % 360.0
                held_candidate = best[2]
                held_since = frame_i
                grace_left = grace_frames
            else:
                held_candidate = None
                grace_left = 0
                yaw = _natural_view_yaw(
                    p,
                    tangent_yaw,
                    look_candidates=None,
                    center_xz=center_xz,
                    max_turn_degrees=look_max_turn_degrees,
                    center_max_turn_degrees=look_center_max_turn_degrees,
                    min_distance=look_min_distance,
                    max_distance=look_max_distance,
                    object_strength=look_object_strength,
                    center_strength=look_center_strength,
                )
        last_yaw = yaw
        poses.append((float(p[0]), float(p[1]), yaw))
    return _stabilize_view_yaw(
        poses,
        fps,
        max_yaw_rate=look_max_yaw_rate,
        smoothing_frames=look_smoothing_frames,
        deadband_degrees=look_deadband_degrees,
        filter_mode=look_filter,
        kalman_process_noise=look_kalman_process_noise,
        kalman_measurement_noise=look_kalman_measurement_noise,
        kalman_initial_uncertainty=look_kalman_initial_uncertainty,
    )


def poses_along_thor_waypoints(controller, positions, objects, fps: int,
                               speed: float, lookahead: float = 0.35,
                               object_types=None, max_waypoints=6,
                               allowed_error=0.25,
                               waypoint_min_clearance=0.35,
                               waypoint_min_open_neighbors=3,
                               look_object_types=None,
                               look_max_turn_degrees=125.0,
                               look_center_max_turn_degrees=180.0,
                               look_min_distance=0.55,
                               look_max_distance=5.0,
                               look_object_strength=0.9,
                               look_center_strength=0.65,
                               look_max_yaw_rate=40.0,
                               look_smoothing_frames=4,
                               look_deadband_degrees=0.0,
                               look_target_hold_frames=60,
                               look_release_grace_frames=8,
                               look_switch_margin=0.15,
                               look_filter="lowpass",
                               look_kalman_process_noise=900.0,
                               look_kalman_measurement_noise=64.0,
                               look_kalman_initial_uncertainty=100.0):
    """Use THOR's shortest-path API between selected room-tour waypoints."""
    ctx = _reachable_context(positions)
    waypoints = select_tour_waypoints(
        positions,
        objects,
        object_types=object_types,
        max_waypoints=max_waypoints,
        min_clearance=waypoint_min_clearance,
        min_open_neighbors=waypoint_min_open_neighbors,
        ctx=ctx,
    )
    if len(waypoints) < 2:
        raise ValueError("not enough tour waypoints to build a THOR path")

    corners = []
    for i, start in enumerate(waypoints):
        goal = waypoints[(i + 1) % len(waypoints)]
        segment = _shortest_path_corners(
            controller, start, goal, allowed_error=allowed_error)
        if corners:
            segment = segment[1:]
        corners.extend(segment)

    corners = _dedupe_positions(corners)
    return _poses_from_polyline_points(
        corners,
        fps,
        speed,
        lookahead,
        look_candidates=_look_candidates(objects, look_object_types),
        center_xz=ctx["center"],
        look_max_turn_degrees=look_max_turn_degrees,
        look_center_max_turn_degrees=look_center_max_turn_degrees,
        look_min_distance=look_min_distance,
        look_max_distance=look_max_distance,
        look_object_strength=look_object_strength,
        look_center_strength=look_center_strength,
        look_max_yaw_rate=look_max_yaw_rate,
        look_smoothing_frames=look_smoothing_frames,
        look_deadband_degrees=look_deadband_degrees,
        look_target_hold_frames=look_target_hold_frames,
        look_release_grace_frames=look_release_grace_frames,
        look_switch_margin=look_switch_margin,
        look_filter=look_filter,
        look_kalman_process_noise=look_kalman_process_noise,
        look_kalman_measurement_noise=look_kalman_measurement_noise,
        look_kalman_initial_uncertainty=look_kalman_initial_uncertainty,
    )


def poses_along_skeleton_exploration(positions, objects, fps: int,
                                     speed: float, lookahead: float = 0.35,
                                     seed=0, coverage=0.65,
                                     prune_length=0.5, smoothing=0.05,
                                     look_object_types=None,
                                     look_max_turn_degrees=125.0,
                                     look_center_max_turn_degrees=180.0,
                                     look_min_distance=0.55,
                                     look_max_distance=5.0,
                                     look_object_strength=0.9,
                                     look_center_strength=0.65,
                                     look_max_yaw_rate=40.0,
                                     look_smoothing_frames=4,
                                     look_deadband_degrees=0.0,
                                     look_target_hold_frames=60,
                                     look_release_grace_frames=8,
                                     look_switch_margin=0.15,
                                     look_filter="lowpass",
                                     look_kalman_process_noise=900.0,
                                     look_kalman_measurement_noise=64.0,
                                     look_kalman_initial_uncertainty=100.0):
    """Generate poses on one fixed skeleton-based exploration loop."""
    ctx = _reachable_context(positions)
    poly = skeleton_exploration_polyline(
        positions,
        seed=seed,
        coverage=coverage,
        prune_length=prune_length,
        smoothing=smoothing,
    )
    points = [
        {"x": float(x), "y": 0.0, "z": float(z)}
        for x, z in poly
    ]
    return _poses_from_polyline_points(
        points,
        fps,
        speed,
        lookahead,
        look_candidates=_look_candidates(objects, look_object_types),
        center_xz=ctx["center"],
        look_max_turn_degrees=look_max_turn_degrees,
        look_center_max_turn_degrees=look_center_max_turn_degrees,
        look_min_distance=look_min_distance,
        look_max_distance=look_max_distance,
        look_object_strength=look_object_strength,
        look_center_strength=look_center_strength,
        look_max_yaw_rate=look_max_yaw_rate,
        look_smoothing_frames=look_smoothing_frames,
        look_deadband_degrees=look_deadband_degrees,
        look_target_hold_frames=look_target_hold_frames,
        look_release_grace_frames=look_release_grace_frames,
        look_switch_margin=look_switch_margin,
        look_filter=look_filter,
        look_kalman_process_noise=look_kalman_process_noise,
        look_kalman_measurement_noise=look_kalman_measurement_noise,
        look_kalman_initial_uncertainty=look_kalman_initial_uncertainty,
    )


def poses_along_skeleton_navigation_loop(positions, objects, fps: int, speed: float,
                                         lookahead: float = 0.35, seed=0,
                                         clearance=0.5, smoothing=0.05,
                                         max_yaw_rate=60.0,
                                         turn_smoothing_distance=0.35,
                                         functional_zone_max_distance=1.6,
                                         functional_zone_merge_distance=1.25):
    """Generate a forward-facing, non-backtracking loop for repeated laps."""
    poly = skeleton_navigation_loop_polyline(
        positions,
        objects=objects,
        seed=seed,
        clearance=clearance,
        smoothing=smoothing,
        functional_zone_max_distance=functional_zone_max_distance,
        functional_zone_merge_distance=functional_zone_merge_distance,
    )
    return _poses_from_closed_polyline(
        poly,
        fps=fps,
        speed=speed,
        lookahead=lookahead,
        max_yaw_rate=max_yaw_rate,
        turn_smoothing_distance=turn_smoothing_distance,
    )


def poses_along_reachable_perimeter(positions, fps: int, speed: float,
                                    lookahead: float = 0.35,
                                    anchor_count: int = 24):
    """Return poses along a collision-respecting loop through reachable cells."""
    poly = _reachable_perimeter_polyline(positions, anchor_count=anchor_count)
    poly, cum_len = _polyline_cumulative(poly)
    step = max(speed / fps, 1e-4)
    total = float(cum_len[-1])
    n = max(2, int(round(total / step)))
    targets = np.linspace(0.0, total, n, endpoint=False)

    poses = []
    yaw_lookahead = max(lookahead, step * 4)
    for d in targets:
        p, _ = _interp_polyline(poly, cum_len, d)
        nxt, _ = _interp_polyline(poly, cum_len, min(d + yaw_lookahead, total))
        if float(np.linalg.norm(nxt - p)) < 1e-6:
            nxt, _ = _interp_polyline(poly, cum_len, max(d - yaw_lookahead, 0.0))
            tangent = p - nxt
        else:
            tangent = nxt - p
        yaw = heading_to(float(tangent[0]), float(tangent[1]))
        poses.append((float(p[0]), float(p[1]), yaw))
    return poses


def poses_along_spline(positions, fps: int, speed: float,
                       inset: float = 0.4, smooth: float = 0.0,
                       lookahead: float = 0.35):
    """Return a list of (x, z, yaw) for every rendered frame of one smooth loop.

    Points are interpolated at equal arc-length spacing (speed / fps meters
    apart), so travel speed is constant with no dense-sample quantization. Yaw
    faces a short look-ahead point on the path, which damps tiny tangent jitter
    around spline corners.
    """
    xy, cum_len = spline_loop(positions, inset=inset, smooth=smooth)
    step = max(speed / fps, 1e-4)
    _, total = _interp_closed(xy, cum_len, 0.0)
    n = max(2, int(round(total / step)))
    targets = np.linspace(0.0, total, n, endpoint=False)

    poses = []
    yaw_lookahead = max(lookahead, step * 4)
    for d in targets:
        p, _ = _interp_closed(xy, cum_len, d)
        nxt, _ = _interp_closed(xy, cum_len, d + yaw_lookahead)
        tangent = nxt - p
        yaw = heading_to(float(tangent[0]), float(tangent[1]))
        poses.append((float(p[0]), float(p[1]), yaw))
    return poses


def _smoothstep(t):
    """C1 ease in/out on [0,1]: zero velocity at both ends (natural head turn)."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def inject_gaze(poses, gaze_points, fps, hold_frames=10, blend_frames=20,
                max_yaw_rate=90.0, max_turn_degrees=None):
    """Insert a slow, smooth "look at it" glance toward each gaze point.

    Around the loop pose closest to each gaze point we rewrite yaw to ease from
    the path tangent to face the point, hold, then ease back -- position keeps
    advancing normally. Smoothness comes from two things:
      * smoothstep easing over `blend_frames` (zero angular velocity at ends), and
      * a hard cap of `max_yaw_rate` deg/s on frame-to-frame yaw change, so even a
        near-180 turn is spread over enough frames to never look snappy.

    Longer blend_frames / lower max_yaw_rate => slower, smoother glance.

    Returns (new_poses, gaze_frames): gaze_frames maps gaze-point index -> the
    frame index at the glance center. Injection only rewrites yaw, so frame count
    is unchanged.
    """
    poses = list(poses)
    n = len(poses)
    gaze_frames = {}
    half = blend_frames + hold_frames // 2
    for gi, (gx, gz) in enumerate(gaze_points):
        c = min(range(n), key=lambda k: (poses[k][0] - gx) ** 2 + (poses[k][1] - gz) ** 2)
        gaze_frames[gi] = c
        for off in range(-half, half + 1):
            k = c + off
            if k < 0 or k >= n:
                continue
            x, z, tangent_yaw = poses[k]
            face_yaw = heading_to(gx - x, gz - z)
            d = abs(off)
            if d <= hold_frames // 2:
                w = 1.0
            else:
                t = (d - hold_frames // 2) / max(blend_frames, 1)
                w = _smoothstep(1.0 - t)  # eased blend, zero-velocity ends
            delta = shortest_delta(tangent_yaw, face_yaw)
            if max_turn_degrees is not None:
                limit = abs(float(max_turn_degrees))
                delta = max(-limit, min(limit, delta))
            yaw = tangent_yaw + delta * w
            poses[k] = (x, z, yaw % 360.0)

    # Enforce a global per-frame yaw-rate cap: no frame may turn more than
    # max_yaw_rate/fps degrees from the previous one. This bounds pan speed
    # everywhere (glances AND lap seams), which is the real smoothness guarantee.
    max_step = max_yaw_rate / fps
    for k in range(1, n):
        px, pz, pyaw = poses[k - 1]
        x, z, yaw = poses[k]
        delta = shortest_delta(pyaw, yaw)
        if abs(delta) > max_step:
            yaw = (pyaw + max_step * (1 if delta > 0 else -1)) % 360.0
            poses[k] = (x, z, yaw)
    return poses, gaze_frames


def cap_yaw_rate(poses, fps, max_yaw_rate=70.0, wrap=True):
    """Bound frame-to-frame yaw change across a pose list.

    Applied to the FULL concatenated multi-lap list so lap seams are smoothed
    too (the per-lap cap inside inject_gaze can't see the seam). With wrap=True
    the last->first transition is also bounded, keeping a looping video seamless.
    """
    poses = list(poses)
    n = len(poses)
    if n < 2:
        return poses
    max_step = max_yaw_rate / fps
    rng = range(1, n) if not wrap else range(1, n + 1)
    for j in rng:
        k = j % n
        pk = (j - 1) % n
        px, pz, pyaw = poses[pk]
        x, z, yaw = poses[k]
        delta = shortest_delta(pyaw, yaw)
        if abs(delta) > max_step:
            yaw = (pyaw + max_step * (1 if delta > 0 else -1)) % 360.0
            poses[k] = (x, z, yaw)
    return poses


def shortest_delta(a: float, b: float) -> float:
    """Signed smallest angular difference b - a in (-180, 180]."""
    return (b - a + 180.0) % 360.0 - 180.0
