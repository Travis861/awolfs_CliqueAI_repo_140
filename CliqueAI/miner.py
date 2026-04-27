import time
import typing

import bittensor as bt
from CliqueAI.graph.codec import GraphCodec
from CliqueAI.protocol import MaximumCliqueOfLambdaGraph
from common.base.miner import BaseMinerNeuron


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


def _greedy_clique_from_candidates(
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


def solve_maximal_clique(
    number_of_nodes: int,
    adjacency_list: list[list[int]],
    timeout: float | None = None,
) -> list[int]:
    if number_of_nodes <= 0:
        return []

    start = time.monotonic()
    if timeout is None or timeout <= 0:
        budget = 8.0
    else:
        budget = max(1.0, min(float(timeout) * 0.82, float(timeout) - 0.35))

    neighbor_masks = _build_neighbor_masks(number_of_nodes, adjacency_list)
    degrees = [mask.bit_count() for mask in neighbor_masks]
    all_vertices = (1 << number_of_nodes) - 1
    ordered_vertices = sorted(
        range(number_of_nodes),
        key=lambda v: degrees[v],
        reverse=True,
    )

    best: list[int] = []
    attempt = 0

    while time.monotonic() - start < budget:
        if attempt < len(ordered_vertices):
            first = ordered_vertices[attempt]
        else:
            first = ordered_vertices[(attempt * 17 + 23) % len(ordered_vertices)]

        initial_candidates = all_vertices & neighbor_masks[first]
        clique = [first] + _greedy_clique_from_candidates(
            initial_candidates,
            neighbor_masks,
            degrees,
            seed_shift=attempt,
        )
        clique = _extend_to_maximal_clique(clique, neighbor_masks)

        if len(clique) > len(best):
            best = clique
            bt.logging.info(f"New best clique size: {len(best)}")

        attempt += 1

        if attempt >= max(number_of_nodes * 3, 256) and len(best) > 0:
            break

    if not best:
        best = _extend_to_maximal_clique([ordered_vertices[0]], neighbor_masks)

    return best


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
