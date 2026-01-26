from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, final

if TYPE_CHECKING:
    from decent_bench.agents import Agent


class DecAlgorithm[NetworkT](ABC):
    """Base class for decentralized algorithms."""

    @property
    @abstractmethod
    def iterations(self) -> int:
        """Number of iterations or rounds to run the algorithm for."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the algorithm."""

    @abstractmethod
    def initialize(self, network: NetworkT) -> None:
        """
        Initialize the algorithm.

        Args:
            network: provides the agents and topology for this algorithm.

        """

    @abstractmethod
    def step(self, network: NetworkT, iteration: int) -> None:
        """
        Perform one iteration or round of the algorithm.

        Args:
            network: provides the agents and topology for this algorithm.
            iteration: current iteration number.

        """

    @abstractmethod
    def _finalize_agents(self, network: NetworkT) -> Iterable["Agent"]:
        """
        Return the agents whose auxiliary variables should be cleared.

        Args:
            network: provides the agents and topology for this algorithm.

        """

    def finalize(self, network: NetworkT) -> None:
        """
        Finalize the algorithm.

        Note:
            Override :meth:`_finalize_agents` to control which agents are finalized.

        Args:
            network: provides the agents and topology for this algorithm.

        """
        for agent in self._finalize_agents(network):
            if agent.aux_vars is not None:
                agent.aux_vars.clear()

    @final
    def run(self, network: NetworkT, progress_callback: Callable[[int], None] | None = None) -> None:
        """
        Run the algorithm.

        Note:
            This method first calls :meth:`initialize`, then :meth:`step` for the specified number of iterations
            and finally :meth:`finalize`.

        Warning:
            Do not override this method. Instead, override :meth:`initialize`, :meth:`step` and :meth:`finalize`
            as needed.

        Args:
            network: provides the agents and topology for this algorithm.
            progress_callback: optional callback to report progress after each iteration.

        """
        self.initialize(network)
        for k in range(self.iterations):
            self.step(network, k)
            if progress_callback is not None:
                progress_callback(k)
        self.finalize(network)
