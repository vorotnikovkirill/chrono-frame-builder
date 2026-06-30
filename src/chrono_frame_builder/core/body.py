from dataclasses import dataclass


@dataclass(frozen=True)
class Body:
    """CAD-derived rigid body metadata."""

    name: str
    cad_file: str
    color: tuple[float, float, float] = (0.7, 0.7, 0.7)
    opacity: float = 1.0
    fixed: bool = False

    @staticmethod
    def from_dict(data: dict) -> "Body":
        """Create a Body from a dictionary."""
        return Body(
            name=data["name"],
            cad_file=data.get("cad_file", ""),
            color=tuple(data.get("color", [0.7, 0.7, 0.7])),
            opacity=float(data.get("opacity", 1.0)),
            fixed=bool(data.get("fixed", False)),
        )

    def to_dict(self) -> dict:
        """Convert a Body to a JSON-serializable dictionary."""
        return {
            "name": self.name,
            "cad_file": self.cad_file,
            "color": list(self.color),
            "opacity": self.opacity,
            "fixed": self.fixed,
        }
