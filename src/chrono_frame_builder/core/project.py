import json
from dataclasses import dataclass
from pathlib import Path

from chrono_frame_builder.core.body import Body
from chrono_frame_builder.core.frame import Frame
from chrono_frame_builder.exceptions import BodyNotFoundError, FrameNotFoundError


@dataclass
class Project:
    """chrono-frame-builder project data model."""

    schema_version: str
    project_name: str
    units: dict
    bodies: list[Body]
    frames: list[Frame]

    @staticmethod
    def load(path: str | Path) -> "Project":
        """Load a project from a JSON file."""
        project_path = Path(path)

        with project_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        bodies = [Body.from_dict(item) for item in data.get("bodies", [])]
        frames = [Frame.from_dict(item) for item in data.get("frames", [])]

        return Project(
            schema_version=data.get("schema_version", "0.1.0"),
            project_name=data.get("project_name", project_path.stem),
            units=data.get("units", {"length": "mm", "angle": "deg"}),
            bodies=bodies,
            frames=frames,
        )

    def save(self, path: str | Path) -> None:
        """Save a project to a JSON file."""
        project_path = Path(path)

        data = {
            "schema_version": self.schema_version,
            "project_name": self.project_name,
            "units": self.units,
            "bodies": [body.to_dict() for body in self.bodies],
            "frames": [frame.to_dict() for frame in self.frames],
        }

        with project_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

            # Keep a newline at the end of the JSON file.
            file.write("\n")

    def list_body_names(self) -> list[str]:
        """Return all body names."""
        return [body.name for body in self.bodies]

    def list_frame_names(self) -> list[str]:
        """Return all frame names in body.frame format."""
        return [frame.full_name for frame in self.frames]

    def get_body(self, name: str) -> Body:
        """Get a body by name."""
        for body in self.bodies:
            if body.name == name:
                return body

        available = ", ".join(self.list_body_names()) or "<none>"
        raise BodyNotFoundError(
            f"Body '{name}' was not found. Available bodies: {available}."
        )

    def get_frame(self, body_name: str, frame_name: str) -> Frame:
        """Get a frame by body name and frame name."""
        for frame in self.frames:
            if frame.body == body_name and frame.name == frame_name:
                return frame

        available = ", ".join(self.list_frame_names()) or "<none>"
        raise FrameNotFoundError(
            f"Frame '{body_name}.{frame_name}' was not found. "
            f"Available frames: {available}."
        )

    def get_frame_by_full_name(self, full_name: str) -> Frame:
        """Get a frame by full name in body.frame format."""
        if "." not in full_name:
            raise FrameNotFoundError(
                f"Frame name '{full_name}' is invalid. Expected format: body.frame."
            )

        body_name, frame_name = full_name.split(".", maxsplit=1)
        return self.get_frame(body_name, frame_name)
