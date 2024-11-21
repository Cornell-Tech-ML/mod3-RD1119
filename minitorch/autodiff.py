from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals_left, vals_right = list(vals), list(vals)
    vals_left[arg] = vals_left[arg] - epsilon
    vals_right[arg] = vals_right[arg] + epsilon
    return (f(*vals_right) - f(*vals_left)) / (2.0 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.


        Args:
        ----
            x: value to be accumulated

        Returns:
        -------
            None

        """
        ...

    @property
    def unique_id(self) -> int:
        """A unique identifier for the variable.

        Args:
        ----
            None

        Returns:
        -------
            int: unique identifier

        """
        ...

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)

        Args:
        ----
            None

        Returns:
        -------
            bool: True if the variable is a leaf variable

        """
        ...

    def is_constant(self) -> bool:
        """True if this variable is a constant

        Args:
        ----
            None

        Returns:
        -------
            bool: True if the variable is a constant

        """
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parents of the variable

        Args:
        ----
            None

        Returns:
        -------
            Iterable["Variable"]: The parents of the variable

        """
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Returns the chain rule of the variable

        Args:
        ----
            d_output: The output

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: The chain rule of the variable

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.
    L = []
    visited = []

    def visit(variable: Variable) -> None:
        if variable.unique_id in visited or variable.is_constant():
            return
        if not variable.is_leaf():
            for node in variable.parents:
                visit(node)
        visited.append(variable.unique_id)
        L.insert(0, variable)

    visit(variable)
    return L


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv: Its derivative that we want to propagate backward to the leaves.

    Returns:
    -------
        None
    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    # TODO: Implement for Task 1.4.
    L = topological_sort(variable)
    derivatives = {item.unique_id: 0 for item in L}
    derivatives[variable.unique_id] = deriv
    for node in L:
        node_drt = derivatives[node.unique_id]
        if node.is_leaf():
            node.accumulate_derivative(node_drt)
        else:
            for input_var, sub_derivative in node.chain_rule(node_drt):
                if input_var.is_constant():
                    continue
                if input_var.unique_id in derivatives:
                    derivatives[input_var.unique_id] += sub_derivative
                else:
                    derivatives[input_var.unique_id] -= sub_derivative


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation.

        Args:
        ----
            *values: The values to be stored.

        Returns:
        -------
            None

        """
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved values.

        Args:
        ----
            None

        Returns:
        -------
            Tuple[Any, ...]: The saved values.

        """
        return self.saved_values
