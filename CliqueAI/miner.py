import time
import typing
from dataclasses import dataclass

import bittensor as bt
from CliqueAI.graph.codec import GraphCodec
from CliqueAI.protocol import MaximumCliqueOfLambdaGraph
from common.base.miner import BaseMinerNeuron


SEARCH_MARGIN_SECONDS = 0.45
DEFAULT_SEARCH_SECONDS = 8.0


def _iter_bits(mask: int):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def _build_neighbor_masks(
    number_of_nodes: int,
    adjacency_list: list[list[int]],
) -> list[int]:
    masks = [0] * number_of_nodes
    for vertex, neighbors in enumerate(adjacency_list):
        mask = 0
        for neighbor in neighbors:
            if 0 <= neighbor < number_of_nodes and neighbor != vertex:
                mask |= 1 << neighbor
        masks[vertex] = mask
    return masks


def _mask_to_sorted_list(mask: int) -> list[int]:
    return sorted(_iter_bits(mask))


def _extend_to_maximal_clique(clique: list[int], neighbor_masks: list[int]) -> list[int]:
    clique_set = set(clique)
    clique_mask = 0
    for vertex in clique_set:
        clique_mask |= 1 << vertex

    changed = True
    while changed:
        changed = False
        for vertex, neighbors in enumerate(neighbor_masks):
            if vertex in clique_set:
                continue
            if clique_mask & ~neighbors == 0:
                clique_set.add(vertex)
                clique_mask |= 1 << vertex
                changed = True
                break
    return sorted(clique_set)


def _greedy_clique(
    candidates: int,
    neighbor_masks: list[int],
    degrees: list[int],
    seed_shift: int,
) -> list[int]:
    clique: list[int] = []

    while candidates:
        vertices = list(_iter_bits(candidates))
        if not vertices:
            break

        vertices.sort(
            key=lambda vertex: (
                (candidates & neighbor_masks[vertex]).bit_count(),
                degrees[vertex],
                -((vertex + seed_shift * 1315423911) % 104729),
            ),
            reverse=True,
        )
        vertex = vertices[0]
        clique.append(vertex)
        candidates &= neighbor_masks[vertex]

    return clique


def _repair_clique(clique: list[int], neighbor_masks: list[int]) -> list[int]:
    repaired: list[int] = []
    candidate_mask = (1 << len(neighbor_masks)) - 1
    for vertex in clique:
        bit = 1 << vertex
        if candidate_mask & bit:
            repaired.append(vertex)
            candidate_mask &= neighbor_masks[vertex]
    return _extend_to_maximal_clique(repaired, neighbor_masks)


@dataclass
class CliqueSearchState:
    deadline: float
    neighbor_masks: list[int]
    degrees: list[int]
    best: list[int]
    timed_out: bool = False

    def update_best(self, clique: list[int]) -> None:
        if len(clique) > len(self.best):
            self.best = clique.copy()
            bt.logging.info(f"New best clique size: {len(self.best)}")

    def has_time(self) -> bool:
        if time.monotonic() >= self.deadline:
            self.timed_out = True
            return False
        return True


def _color_sort(candidates: int, state: CliqueSearchState) -> tuple[list[int], list[int]]:
    vertices: list[int] = []
    color_bounds: list[int] = []
    uncolored = candidates
    color = 0

    while uncolored:
        color += 1
        available = uncolored
        while available:
            best_vertex = max(
                _iter_bits(available),
                key=lambda v: (
                    (available & state.neighbor_masks[v]).bit_count(),
                    state.degrees[v],
                ),
            )
            vertices.append(best_vertex)
            color_bounds.append(color)
            bit = 1 << best_vertex
            uncolored &= ~bit
            available &= ~bit
            available &= ~state.neighbor_masks[best_vertex]

    return vertices, color_bounds


def _expand_max_clique(
    current: list[int],
    candidates: int,
    state: CliqueSearchState,
) -> None:
    if not candidates:
        state.update_best(current)
        return

    if not state.has_time():
        return

    ordered_vertices, color_bounds = _color_sort(candidates, state)

    for index in range(len(ordered_vertices) - 1, -1, -1):
        if not state.has_time():
            return

        if len(current) + color_bounds[index] <= len(state.best):
            return

        vertex = ordered_vertices[index]
        bit = 1 << vertex
        if not candidates & bit:
            continue

        next_candidates = candidates & state.neighbor_masks[vertex]
        current.append(vertex)

        if next_candidates:
            _expand_max_clique(current, next_candidates, state)
        else:
            state.update_best(current)

        current.pop()
        candidates &= ~bit


def _seed_with_greedy_search(
    all_vertices: int,
    neighbor_masks: list[int],
    degrees: list[int],
    deadline: float,
) -> list[int]:
    ordered_vertices = sorted(
        range(len(neighbor_masks)),
        key=lambda v: degrees[v],
        reverse=True,
    )
    best: list[int] = []
    attempt = 0
    max_attempts = max(len(neighbor_masks) * 2, 128)

    while attempt < max_attempts and time.monotonic() < deadline:
        if attempt < len(ordered_vertices):
            first = ordered_vertices[attempt]
        else:
            first = ordered_vertices[(attempt * 17 + 23) % len(ordered_vertices)]

        clique = [first] + _greedy_clique(
            all_vertices & neighbor_masks[first],
            neighbor_masks,
            degrees,
            seed_shift=attempt,
        )
        clique = _repair_clique(clique, neighbor_masks)

        if len(clique) > len(best):
            best = clique
            bt.logging.info(f"Greedy seed clique size: {len(best)}")

        attempt += 1

    return best


def solve_maximal_clique(
    number_of_nodes: int,
    adjacency_list: list[list[int]],
    timeout: float | None = None,
) -> list[int]:
    if number_of_nodes <= 0:
        return []

    start = time.monotonic()
    if timeout is None or timeout <= 0:
        budget = DEFAULT_SEARCH_SECONDS
    else:
        budget = max(
            1.0,
            min(float(timeout) * 0.9, float(timeout) - SEARCH_MARGIN_SECONDS),
        )
    deadline = start + budget

    neighbor_masks = _build_neighbor_masks(number_of_nodes, adjacency_list)
    degrees = [mask.bit_count() for mask in neighbor_masks]
    all_vertices = (1 << number_of_nodes) - 1

    seed_deadline = min(deadline, start + max(0.2, budget * 0.15))
    best = _seed_with_greedy_search(
        all_vertices=all_vertices,
        neighbor_masks=neighbor_masks,
        degrees=degrees,
        deadline=seed_deadline,
    )

    search_state = CliqueSearchState(
        deadline=deadline,
        neighbor_masks=neighbor_masks,
        degrees=degrees,
        best=best,
    )
    _expand_max_clique([], all_vertices, search_state)
    best = search_state.best

    if not best:
        best_vertex = max(range(number_of_nodes), key=lambda v: degrees[v])
        best = _extend_to_maximal_clique([best_vertex], neighbor_masks)

    return _mask_to_sorted_list(sum(1 << vertex for vertex in _repair_clique(best, neighbor_masks)))


class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    def __init__(self, config=None):
        super().__init__(config=config)
        self.axon.attach(
            forward_fn=self.forward_graph,
            blacklist_fn=self.backlist_graph,
            priority_fn=self.priority_graph,
        )

    async def forward_graph(
        self, synapse: MaximumCliqueOfLambdaGraph
    ) -> MaximumCliqueOfLambdaGraph:
        started_at = time.monotonic()
        try:
            codec = GraphCodec()
            adjacency_matrix = codec.decode_matrix(synapse.encoded_matrix)
            adjacency_list = codec.matrix_to_list(adjacency_matrix)
            maximum_clique = solve_maximal_clique(
                number_of_nodes=synapse.number_of_nodes,
                adjacency_list=adjacency_list,
                timeout=getattr(synapse, "timeout", None),
            )
        except Exception as exc:
            bt.logging.error(f"Failed to solve clique request {synapse.uuid}: {exc}")
            maximum_clique = []

        bt.logging.info(
            "Clique response prepared | "
            f"uuid={synapse.uuid} nodes={synapse.number_of_nodes} "
            f"size={len(maximum_clique)} elapsed={time.monotonic() - started_at:.3f}s"
        )
        synapse.maximum_clique = maximum_clique
        return synapse

    async def backlist_graph(
        self, synapse: MaximumCliqueOfLambdaGraph
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority_graph(self, synapse: MaximumCliqueOfLambdaGraph) -> float:
        return await self.priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Miner has started running.")
        while True:
            if miner.should_exit:
                bt.logging.info("Miner is exiting.")
                break
            time.sleep(1)
