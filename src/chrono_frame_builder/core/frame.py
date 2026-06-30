from dataclasses import dataclass

import numpy as np

from chrono_frame_builder.exceptions import InvalidFrameError


@dataclass(frozen=True)
class Frame:
    """Body-attached coordinate frame."""

    body: str
    name: str
    origin: np.ndarray
    rotation_matrix: np.ndarray

    @staticmethod
    def from_dict(data: dict) -> "Frame":
        """Create a Frame from a dictionary."""
        origin = np.array(data["origin"], dtype=float)
        rotation_matrix = np.array(data["rotation_matrix"], dtype=float)

        frame = Frame(
            body=data["body"],
            name=data["name"],
            origin=origin,
            rotation_matrix=rotation_matrix,
        )
        frame.validate()
        return frame

    @property
    def full_name(self) -> str:
        """Return frame name in body.frame format."""
        return f"{self.body}.{self.name}"

    def validate(self) -> None:
        """Validate frame dimensions."""
        if self.origin.shape != (3,):
            raise InvalidFrameError(
                f"Frame '{self.full_name}' origin must have shape (3,), got {self.origin.shape}."
            )

        if self.rotation_matrix.shape != (3, 3):
            raise InvalidFrameError(
                f"Frame '{self.full_name}' rotation_matrix must have shape (3, 3), "
                f"got {self.rotation_matrix.shape}."
            )

    def to_dict(self) -> dict:
        """Convert a Frame to a JSON-serializable dictionary."""
        return {
            "body": self.body,
            "name": self.name,
            "origin": self.origin.tolist(),
            "rotation_matrix": self.rotation_matrix.tolist(),
        }
