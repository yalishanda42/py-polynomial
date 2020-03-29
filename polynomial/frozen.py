"""This module defines the Freezable interface."""


class Freezable:
    """An interface for freezable objects."""

    def _freeze(self):
        """Prevent further modification of self."""
        if not self._is_frozen():
            self._frozen = True

    def _is_frozen(self):
        """Return true if self is frozen."""
        return getattr(self, "_frozen", False)

    def __setitem__(self, key, value):
        """Implement self[x] = y; disallows setting item if frozen."""
        if self._is_frozen():
            raise AttributeError()
        super().__setitem__(self, key, value)

    def __setattr__(self, key, value):
        """Implement self.x; disallows setting attr if frozen."""
        if not self._is_frozen():
            object.__setattr__(self, key, value)
        else:
            raise AttributeError("Can not modify frozen object.")
