import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, final

import decent_bench.utils.algorithm_helpers as alg_helpers
import decent_bench.utils.interoperability as iop
from decent_bench.networks import FedNetwork
from decent_bench.schemes import ClientSelectionScheme, UniformClientSelection

if TYPE_CHECKING:
    from decent_bench.agents import Agent
    from decent_bench.utils.array import Array


class Algorithm(ABC):
    """Federated algorithm - clients collaborate via a central server."""

    @property
    @abstractmethod
    def iterations(self) -> int:
        """Number of rounds to run the algorithm for."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the algorithm."""

    @abstractmethod
    def initialize(self, network: FedNetwork) -> None:
        """
        Initialize the algorithm.

        Args:
            network: provides clients and server

        """

    @abstractmethod
    def step(self, network: FedNetwork, iteration: int) -> None:
        """
        Perform one round of the algorithm.

        Args:
            network: provides clients and server
            iteration: current round number

        """

    def finalize(self, network: FedNetwork) -> None:
        """
        Finalize the algorithm.

        Note:
            Override method as needed.
            Does not need to be implemented if no finalization is required.
            By default it is used to clean up auxiliary variables to free memory.

        Args:
            network: provides clients and server

        """
        for agent in [network.server, *network.clients]:
            if agent.aux_vars is not None:
                agent.aux_vars.clear()

    @final
    def run(self, network: FedNetwork, progress_callback: Callable[[int], None] | None = None) -> None:
        """
        Run the algorithm.

        Note:
            This method first calls :meth:`initialize`, then :meth:`step` for the specified number of iterations
            and finally :meth:`finalize`.

        Warning:
            Do not override this method. Instead, override :meth:`initialize`, :meth:`step` and :meth:`finalize`
            as needed.

        Args:
            network: provides clients and server
            progress_callback: optional callback to report progress after each round.

        """
        self.initialize(network)
        for k in range(self.iterations):
            self.step(network, k)
            if progress_callback is not None:
                progress_callback(k)
        self.finalize(network)

    @staticmethod
    def _select_clients(
        clients: Sequence["Agent"],
        iteration: int,
        selection_scheme: ClientSelectionScheme | None,
    ) -> list["Agent"]:
        if selection_scheme is None:
            return list(clients)
        return selection_scheme.select(clients, iteration)

    @staticmethod
    def _infer_client_weight(client: "Agent") -> float:
        cost = client.cost
        if hasattr(cost, "A"):
            try:
                size = iop.shape(cost.A)[0]
            except Exception:
                size = None
            if size is not None:
                return float(size)
        if hasattr(cost, "b"):
            try:
                size = iop.shape(cost.b)[0]
            except Exception:
                size = None
            if size is not None:
                return float(size)
        if hasattr(cost, "n_samples"):
            n_samples = cost.n_samples
            if n_samples is not None:
                return float(n_samples)
        raise ValueError(
            "Cannot infer client data size. Provide client_weights to the algorithm or add a size "
            "attribute to the cost."
        )

    @classmethod
    def _weights_for_clients(
        cls,
        clients: Sequence["Agent"],
        client_weights: dict[int, float] | Sequence[float] | None,
    ) -> list[float]:
        if client_weights is None:
            weights = [cls._infer_client_weight(client) for client in clients]
        elif isinstance(client_weights, dict):
            weights = []
            for client in clients:
                if client.id not in client_weights:
                    raise ValueError(f"Missing weight for client id {client.id}")
                weights.append(float(client_weights[client.id]))
        else:
            max_id = max(client.id for client in clients)
            if len(client_weights) <= max_id:
                raise ValueError("client_weights sequence must be indexed by client id")
            weights = [float(client_weights[client.id]) for client in clients]
        if any(weight < 0 for weight in weights):
            raise ValueError("Client weights must be non-negative")
        return weights


@dataclass(eq=False)
class FedAvg(Algorithm):
    r"""
    Federated Averaging (FedAvg) with local SGD epochs.

    .. math::
        \mathbf{x}_{i, k}^{(t+1)} = \mathbf{x}_{i, k}^{(t)} - \eta \nabla f_i(\mathbf{x}_{i, k}^{(t)})
    .. math::
        \mathbf{x}_{k+1} = \frac{1}{|S_k|} \sum_{i \in S_k} \mathbf{x}_{i, k}^{(E)}

    where :math:`E` is the number of local epochs per round, :math:`\eta` is the step size, and :math:`S_k` is the set
    of participating clients at round :math:`k`. The aggregation uses client weights, defaulting to data-size weights
    when ``client_weights`` is not provided. Client selection (subsampling) defaults to uniform sampling with
    fraction 1.0 (all active clients) and can be customized via ``selection_scheme``. Local updates use stochastic
    gradients computed over all minibatches of size ``batch_size`` per epoch. Set ``batch_size=None`` to use
    full-batch gradients.

    """

    # C=0.1; batch size= inf/10/50 (dataset sizes are bigger; normally 1/10 of the total dataset).
    # E= 5/20 (num local epochs).
    step_size: float
    local_epochs: int = 1
    batch_size: int | None = 1
    sgd_seed: int | None = None
    client_weights: dict[int, float] | Sequence[float] | None = None
    selection_scheme: ClientSelectionScheme | None = field(
        default_factory=lambda: UniformClientSelection(client_fraction=1.0)
    )
    x0: "Array | None" = None
    iterations: int = 100
    name: str = "FedAvg"

    def initialize(self, network: FedNetwork) -> None:  # noqa: D102
        self.x0 = alg_helpers.zero_initialization(self.x0, network)
        server = network.server
        clients = network.clients
        server.initialize(x=self.x0, received_msgs=dict.fromkeys(clients, self.x0))
        for client in clients:
            client.initialize(x=self.x0, received_msgs={server: self.x0})

    def _get_sgd_rng(self) -> random.Random | None:
        if self.sgd_seed is None:
            return None
        if not hasattr(self, "_sgd_rng"):
            self._sgd_rng = random.Random(self.sgd_seed)
        return self._sgd_rng

    def step(self, network: FedNetwork, iteration: int) -> None:  # noqa: D102
        server = network.server
        active_clients = network.active_clients(iteration)
        if not active_clients:
            return
        selected_clients = self._select_clients(active_clients, iteration, self.selection_scheme)
        if not selected_clients:
            return

        network.send(sender=server, receiver=selected_clients, msg=server.x)
        for client in selected_clients:
            network.receive(receiver=client, sender=server)

        for client in selected_clients:
            local_x = client.messages[server]
            if self.batch_size is None:
                for _ in range(self.local_epochs):
                    grad = client.cost.stochastic_gradient(local_x, batch_size=None, rng=self._get_sgd_rng())
                    local_x -= self.step_size * grad
            else:
                n_samples = int(self._infer_client_weight(client))
                if n_samples <= 0:
                    raise ValueError("Client dataset size must be positive")
                rng = self._get_sgd_rng() or random.Random()
                for _ in range(self.local_epochs):
                    indices = list(range(n_samples))
                    rng.shuffle(indices)
                    for start in range(0, n_samples, self.batch_size):
                        batch_indices = indices[start : start + self.batch_size]
                        grad = client.cost.stochastic_gradient(local_x, batch_indices=batch_indices)
                        local_x -= self.step_size * grad
            client.x = local_x
            network.send(sender=client, receiver=server, msg=client.x)

        network.receive(receiver=server, sender=selected_clients)
        updates = [server.messages[client] for client in selected_clients]
        weights = self._weights_for_clients(selected_clients, self.client_weights)
        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("Sum of client weights must be positive")
        weighted_updates = [update * weight for update, weight in zip(updates, weights, strict=True)]
        server.x = iop.sum(iop.stack(weighted_updates, dim=0), dim=0) / total_weight
