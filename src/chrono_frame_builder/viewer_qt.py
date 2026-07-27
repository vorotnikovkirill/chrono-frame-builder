"""Optional integrated Qt UI for preview-only feature-based frame creation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from chrono_frame_builder.core.features import FeatureCandidate, MeshEdgeSnapCandidate
from chrono_frame_builder.core.marker_editor import (
    CreateFrameEditorState,
    create_frame_editor_summary,
)
from chrono_frame_builder.core.project import Project
from chrono_frame_builder.viewer import (
    AXIS_COLORS,
    AXIS_SELECTOR_ORDER,
    _format_vector,
    _remove_named_actors,
    collect_geometry,
    create_frame_actor_names,
    create_frame_preview_style_from_bounds,
    feature_summary,
    frame_axis_segments,
    infer_axis_length,
    project_axis_segments,
    snap_candidate_from_displayable_mesh_pick,
)


def qt_ui_status() -> tuple[bool, str]:
    """Report whether the optional Qt create-frame UI can be imported."""
    try:
        import pyvista  # noqa: F401
        import pyvistaqt  # noqa: F401
        from PySide6 import QtWidgets  # noqa: F401
    except (ImportError, OSError) as error:
        return False, f"Integrated Qt UI is unavailable ({error})."

    return True, "Integrated Qt UI is available."


def render_create_frame_qt(
    project: Project,
    project_path: str | Path,
    *,
    axis_length: float | None,
) -> None:
    """Open one Qt window with a PyVista view and preview marker-list editor."""
    import pyvista as pv
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QApplication,
        QButtonGroup,
        QComboBox,
        QDockWidget,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QPushButton,
        QRadioButton,
        QVBoxLayout,
        QWidget,
    )
    from pyvistaqt import QtInteractor

    app = QApplication.instance()
    owns_application = app is None
    if app is None:
        app = QApplication(sys.argv)

    window = QMainWindow()
    window.setWindowTitle(f"chrono-frame-builder: {project.project_name}")
    plotter = QtInteractor(window)
    window.setCentralWidget(plotter)

    editor_state = CreateFrameEditorState()
    geometry = collect_geometry(project, project_path)
    geometry_bounds = []
    displayable_meshes = []
    for item in geometry:
        if not item.is_displayable:
            continue
        mesh = pv.read(str(item.path))
        plotter.add_mesh(
            mesh,
            color=item.color,
            opacity=item.opacity,
            name=f"geometry_{item.body_name}",
            show_edges=True,
            edge_color="#202020",
            line_width=1,
        )
        geometry_bounds.append(tuple(mesh.bounds))
        displayable_meshes.append((item, mesh))

    resolved_axis_length = axis_length or infer_axis_length(project, geometry_bounds)
    for segment in project_axis_segments(project, axis_length=resolved_axis_length):
        plotter.add_mesh(
            pv.Line(segment.start, segment.end),
            color=segment.color,
            line_width=3,
            name=f"stored_frame_{segment.frame_name}_{segment.axis_name}",
        )
    if project.frames:
        plotter.add_point_labels(
            np.array([frame.origin for frame in project.frames], dtype=float),
            [frame.full_name for frame in project.frames],
            font_size=9,
            point_size=0,
            always_visible=True,
        )

    style = create_frame_preview_style_from_bounds(geometry_bounds)
    panel = QWidget(window)
    panel_layout = QVBoxLayout(panel)
    panel_layout.setContentsMargins(12, 12, 12, 12)
    panel_layout.setSpacing(8)
    title = QLabel("Preview Markers", panel)
    title.setStyleSheet("font-weight: 600; font-size: 14px;")
    panel_layout.addWidget(title)
    new_marker_button = QPushButton("New Marker", panel)
    panel_layout.addWidget(new_marker_button)

    click_mode = QComboBox(panel)
    click_mode.addItem("Create marker", "create_marker")
    click_mode.addItem("Select/Edit marker", "select_edit")
    click_mode.setCurrentIndex(click_mode.findData(editor_state.click_mode))
    panel_layout.addWidget(QLabel("Click mode", panel))
    panel_layout.addWidget(click_mode)
    click_mode_status = QLabel("Active: Create marker", panel)
    click_mode_status.setStyleSheet("font-weight: 600;")
    panel_layout.addWidget(click_mode_status)

    marker_list = QListWidget(panel)
    marker_list.setMinimumHeight(120)
    panel_layout.addWidget(marker_list)

    properties = QGroupBox("Selected Marker Properties", panel)
    properties_layout = QVBoxLayout(properties)
    name_layout = QFormLayout()
    marker_name = QLineEdit(properties)
    name_layout.addRow("Name", marker_name)
    properties_layout.addLayout(name_layout)

    position_layout = QGridLayout()
    position_fields = []
    for column, axis_name in enumerate(("X", "Y", "Z")):
        position_layout.addWidget(QLabel(axis_name, properties), 0, column)
        field = QLineEdit(properties)
        field.setMaximumWidth(92)
        position_layout.addWidget(field, 1, column)
        position_fields.append(field)
    properties_layout.addLayout(position_layout)

    rotation_layout = QGridLayout()
    for column in range(3):
        rotation_layout.addWidget(QLabel(f"R{column + 1}", properties), 0, column + 1)
    rotation_fields = []
    for row in range(3):
        rotation_layout.addWidget(QLabel(f"R{row + 1}", properties), row + 1, 0)
        row_fields = []
        for column in range(3):
            field = QLineEdit(properties)
            field.setMaximumWidth(70)
            field.setToolTip(f"R{row + 1}{column + 1}")
            rotation_layout.addWidget(field, row + 1, column + 1)
            row_fields.append(field)
        rotation_fields.append(row_fields)
    properties_layout.addLayout(rotation_layout)
    apply_properties_button = QPushButton("Apply marker properties", properties)
    properties_layout.addWidget(apply_properties_button)
    panel_layout.addWidget(properties)

    origin_section = QGroupBox("Frame Origin", panel)
    origin_layout = QFormLayout(origin_section)
    origin_source = QLabel(origin_section)
    origin_detail = QLabel("Selected geometry point, edge midpoint, or face center", origin_section)
    origin_detail.setWordWrap(True)
    origin_layout.addRow("Source", origin_source)
    origin_layout.addRow("Available", origin_detail)
    reference_origin = QLabel("Reference origin / center of mass: planned", origin_section)
    reference_origin.setEnabled(False)
    origin_layout.addRow(reference_origin)
    panel_layout.addWidget(origin_section)

    primary_section = QGroupBox("Primary Axis", panel)
    primary_layout = QVBoxLayout(primary_section)
    primary_axis = QComboBox(primary_section)
    primary_axis.addItems(AXIS_SELECTOR_ORDER)
    primary_reference_axis = QComboBox(primary_section)
    primary_reference_axis.addItems(AXIS_SELECTOR_ORDER)
    primary_layout.addWidget(QLabel("Marker axis", primary_section))
    primary_layout.addWidget(primary_axis)
    primary_layout.addWidget(QLabel("Reference/source axis", primary_section))
    primary_layout.addWidget(primary_reference_axis)
    primary_reference = QRadioButton("Along Reference Frame Axis", primary_section)
    primary_inertia = QRadioButton("Along Principal Inertia Axis (planned)", primary_section)
    primary_inertia.setEnabled(False)
    primary_feature = QRadioButton("Based on Geometric Feature", primary_section)
    primary_group = QButtonGroup(primary_section)
    for button in (primary_reference, primary_inertia, primary_feature):
        primary_group.addButton(button)
        primary_layout.addWidget(button)
    primary_use_feature = QPushButton("Pick Primary Feature", primary_section)
    primary_flip_direction = QPushButton("Flip Primary Direction", primary_section)
    primary_source = QLabel("<none>", primary_section)
    primary_layout.addWidget(primary_use_feature)
    primary_layout.addWidget(primary_flip_direction)
    primary_layout.addWidget(primary_source)
    panel_layout.addWidget(primary_section)

    secondary_section = QGroupBox("Secondary Axis", panel)
    secondary_layout = QVBoxLayout(secondary_section)
    secondary_axis = QComboBox(secondary_section)
    secondary_axis.addItems(AXIS_SELECTOR_ORDER)
    secondary_reference_axis = QComboBox(secondary_section)
    secondary_reference_axis.addItems(AXIS_SELECTOR_ORDER)
    secondary_layout.addWidget(QLabel("Marker axis", secondary_section))
    secondary_layout.addWidget(secondary_axis)
    secondary_layout.addWidget(QLabel("Reference/source axis", secondary_section))
    secondary_layout.addWidget(secondary_reference_axis)
    secondary_reference = QRadioButton("Along Reference Frame Axis", secondary_section)
    secondary_inertia = QRadioButton("Along Principal Inertia Axis (planned)", secondary_section)
    secondary_inertia.setEnabled(False)
    secondary_feature = QRadioButton("Based on Geometric Feature", secondary_section)
    secondary_group = QButtonGroup(secondary_section)
    for button in (secondary_reference, secondary_inertia, secondary_feature):
        secondary_group.addButton(button)
        secondary_layout.addWidget(button)
    secondary_use_feature = QPushButton("Pick Secondary Feature", secondary_section)
    secondary_flip_direction = QPushButton("Flip Secondary Direction", secondary_section)
    secondary_source = QLabel("<none>", secondary_section)
    secondary_layout.addWidget(secondary_use_feature)
    secondary_layout.addWidget(secondary_flip_direction)
    secondary_layout.addWidget(secondary_source)
    reset_axes = QPushButton("Reset axes to reference", secondary_section)
    secondary_layout.addWidget(reset_axes)
    panel_layout.addWidget(secondary_section)

    message_label = QLabel("Click New Marker, then click geometry to place marker_001.", panel)
    message_label.setWordWrap(True)
    message_label.setStyleSheet("color: #7f1d1d;")
    panel_layout.addWidget(message_label)
    panel_layout.addStretch(1)

    dock = QDockWidget("Create Frame", window)
    dock.setObjectName("create_frame_dock")
    dock.setWidget(panel)
    dock.setMinimumWidth(370)
    window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    marker_actor_names: list[str] = []

    def feature_source_text(candidate: FeatureCandidate | MeshEdgeSnapCandidate | None) -> str:
        if candidate is None:
            return "<none>"
        if isinstance(candidate, MeshEdgeSnapCandidate):
            return "edge tangent"
        if candidate.direction is not None:
            return "face normal"
        return "<no direction>"

    def refresh_marker_list() -> None:
        marker_list.blockSignals(True)
        marker_list.clear()
        for marker in editor_state.markers:
            item = QListWidgetItem(f"{marker.id:03d}  {marker.name}")
            item.setData(Qt.ItemDataRole.UserRole, marker.id)
            marker_list.addItem(item)
            if marker.id == editor_state.selected_marker_id:
                marker_list.setCurrentItem(item)
        marker_list.blockSignals(False)

    def update_panel(message: str | None = None) -> None:
        marker = editor_state.selected_marker
        summary = create_frame_editor_summary(editor_state)
        message_label.setText(message or summary["message"])
        click_mode_labels = {
            "create_marker": "Create Marker",
            "select_edit": "Select/Edit Marker",
            "pick_primary_vector": "Pick Primary Vector",
            "pick_secondary_vector": "Pick Secondary Vector",
        }
        click_mode_status.setText(
            f"Active: {click_mode_labels.get(editor_state.click_mode, editor_state.click_mode)}"
        )
        click_mode.blockSignals(True)
        if editor_state.click_mode in {"create_marker", "select_edit"}:
            click_mode.setCurrentIndex(click_mode.findData(editor_state.click_mode))
        click_mode.blockSignals(False)
        origin_source.setText(summary["origin_source"])
        properties.setEnabled(marker is not None)
        origin_section.setEnabled(marker is not None)
        primary_section.setEnabled(marker is not None)
        secondary_section.setEnabled(marker is not None)
        if marker is None:
            marker_name.clear()
            for field in [*position_fields, *[item for row in rotation_fields for item in row]]:
                field.clear()
            primary_source.setText("<none>")
            secondary_source.setText("<none>")
            refresh_marker_list()
            return

        marker_name.setText(marker.name)
        for field, value in zip(position_fields, marker.origin, strict=True):
            field.setText(f"{value:.9g}")
        for row, fields in enumerate(rotation_fields):
            for column, field in enumerate(fields):
                field.setText(f"{marker.rotation_matrix[row, column]:.9g}")
        primary_axis.blockSignals(True)
        secondary_axis.blockSignals(True)
        primary_reference_axis.blockSignals(True)
        secondary_reference_axis.blockSignals(True)
        primary_axis.setCurrentText(marker.primary_axis)
        secondary_axis.setCurrentText(marker.secondary_axis)
        primary_reference_axis.setCurrentText(marker.primary_reference_axis)
        secondary_reference_axis.setCurrentText(marker.secondary_reference_axis)
        primary_axis.blockSignals(False)
        secondary_axis.blockSignals(False)
        primary_reference_axis.blockSignals(False)
        secondary_reference_axis.blockSignals(False)
        primary_reference.blockSignals(True)
        primary_feature.blockSignals(True)
        secondary_reference.blockSignals(True)
        secondary_feature.blockSignals(True)
        primary_reference.setChecked(marker.primary_axis_source_mode == "reference")
        primary_feature.setChecked(marker.primary_axis_source_mode == "feature")
        secondary_reference.setChecked(marker.secondary_axis_source_mode == "reference")
        secondary_feature.setChecked(marker.secondary_axis_source_mode == "feature")
        primary_reference.blockSignals(False)
        primary_feature.blockSignals(False)
        secondary_reference.blockSignals(False)
        secondary_feature.blockSignals(False)
        primary_reference_axis.setEnabled(marker.primary_axis_source_mode == "reference")
        secondary_reference_axis.setEnabled(marker.secondary_axis_source_mode == "reference")
        primary_source.setText(feature_source_text(marker.primary_feature))
        secondary_source.setText(feature_source_text(marker.secondary_feature))
        refresh_marker_list()

    def report_error(error: ValueError) -> None:
        editor_state.warning = str(error)
        print(f"warning: {error}", file=sys.stderr)
        update_panel()

    def clear_selected_feature_cue() -> None:
        _remove_named_actors(plotter, create_frame_actor_names("qt_selected"))

    def draw_selected_feature(candidate: FeatureCandidate | MeshEdgeSnapCandidate) -> None:
        """Draw a thin cue only while choosing a geometric axis source."""
        clear_selected_feature_cue()
        if isinstance(candidate, MeshEdgeSnapCandidate):
            plotter.add_mesh(
                pv.Line(candidate.start, candidate.end),
                color="orange",
                line_width=2,
                name="qt_selected_edge",
            )
        else:
            marker = pv.Sphere(radius=style.marker_radius * 0.55, center=candidate.point)
            plotter.add_mesh(marker, color="yellow", name="qt_selected_point")
        plotter.render()

    def draw_markers() -> None:
        _remove_named_actors(plotter, marker_actor_names)
        marker_actor_names.clear()
        label_points = []
        label_text = []
        for marker in editor_state.markers:
            frame = marker.as_frame()
            line_width = 6 if marker.id == editor_state.selected_marker_id else 4
            for segment in frame_axis_segments(frame, axis_length=resolved_axis_length):
                actor_name = f"preview_marker_{marker.id}_{segment.axis_name}"
                plotter.add_mesh(
                    pv.Line(segment.start, segment.end),
                    color=AXIS_COLORS[segment.axis_name],
                    line_width=line_width,
                    name=actor_name,
                )
                marker_actor_names.append(actor_name)
            origin_name = f"preview_marker_{marker.id}_origin"
            origin_scale = 0.03 if marker.id == editor_state.selected_marker_id else 0.02
            radius = resolved_axis_length * origin_scale
            plotter.add_mesh(
                pv.Sphere(radius=radius, center=marker.origin),
                color="white",
                name=origin_name,
            )
            marker_actor_names.append(origin_name)
            label_points.append(marker.origin)
            label_text.append(marker.name)
        if label_points:
            try:
                plotter.add_point_labels(
                    np.asarray(label_points),
                    label_text,
                    name="preview_marker_labels",
                    font_size=10,
                    point_size=0,
                    always_visible=True,
                )
                marker_actor_names.append("preview_marker_labels")
            except TypeError:
                plotter.add_point_labels(
                    np.asarray(label_points),
                    label_text,
                    font_size=10,
                    point_size=0,
                    always_visible=True,
                )
        plotter.render()

    def apply_properties() -> None:
        try:
            origin = np.asarray([float(field.text()) for field in position_fields], dtype=float)
            rotation_matrix = np.asarray(
                [[float(field.text()) for field in row] for row in rotation_fields],
                dtype=float,
            )
        except ValueError:
            report_error(ValueError("Marker property values must be valid finite numbers."))
            return
        try:
            editor_state.rename_selected_marker(marker_name.text())
            editor_state.apply_selected_transform(origin, rotation_matrix)
        except ValueError as error:
            report_error(error)
            return
        draw_markers()
        update_panel("Marker properties applied. Preview only: no project files are changed.")

    def select_list_marker(item) -> None:
        if item is None:
            return
        try:
            editor_state.select_marker(int(item.data(Qt.ItemDataRole.UserRole)))
        except ValueError as error:
            report_error(error)
            return
        draw_markers()
        update_panel()

    def set_click_mode(_index: int) -> None:
        editor_state.set_click_mode(str(click_mode.currentData()))
        update_panel()

    def begin_new_marker() -> None:
        editor_state.begin_new_marker()
        update_panel("Click geometry to place the next preview marker.")

    def set_primary_axis(axis_selector: str) -> None:
        try:
            editor_state.set_primary_axis(axis_selector)
        except ValueError as error:
            report_error(error)
            return
        draw_markers()
        update_panel()

    def set_secondary_axis(axis_selector: str) -> None:
        try:
            editor_state.set_secondary_axis(axis_selector)
        except ValueError as error:
            report_error(error)
            return
        draw_markers()
        update_panel()

    def set_primary_reference_axis(axis_selector: str) -> None:
        try:
            editor_state.set_primary_reference_axis(axis_selector)
        except ValueError as error:
            report_error(error)
            return
        draw_markers()
        update_panel()

    def set_secondary_reference_axis(axis_selector: str) -> None:
        try:
            editor_state.set_secondary_reference_axis(axis_selector)
        except ValueError as error:
            report_error(error)
            return
        draw_markers()
        update_panel()

    def set_primary_mode(use_feature: bool) -> None:
        if not use_feature:
            return
        try:
            editor_state.set_primary_axis_source_mode("feature")
        except ValueError as error:
            report_error(error)
            return
        update_panel()

    def set_primary_reference_mode(use_reference: bool) -> None:
        if not use_reference:
            return
        try:
            editor_state.set_primary_axis_source_mode("reference")
        except ValueError as error:
            report_error(error)
            return
        draw_markers()
        update_panel()

    def set_secondary_mode(use_feature: bool) -> None:
        if not use_feature:
            return
        try:
            editor_state.set_secondary_axis_source_mode("feature")
        except ValueError as error:
            report_error(error)
            return
        update_panel()

    def set_secondary_reference_mode(use_reference: bool) -> None:
        if not use_reference:
            return
        try:
            editor_state.set_secondary_axis_source_mode("reference")
        except ValueError as error:
            report_error(error)
            return
        draw_markers()
        update_panel()

    def begin_pick_primary_feature() -> None:
        try:
            editor_state.begin_pick_primary_vector()
        except ValueError as error:
            report_error(error)
            return
        update_panel("Pick a face normal or edge tangent for the primary axis.")

    def begin_pick_secondary_feature() -> None:
        try:
            editor_state.begin_pick_secondary_vector()
        except ValueError as error:
            report_error(error)
            return
        update_panel("Pick a face normal or edge tangent for the secondary axis.")

    def flip_primary_direction() -> None:
        try:
            editor_state.flip_primary_direction()
        except ValueError as error:
            report_error(error)
            return
        draw_markers()
        update_panel()

    def flip_secondary_direction() -> None:
        try:
            editor_state.flip_secondary_direction()
        except ValueError as error:
            report_error(error)
            return
        draw_markers()
        update_panel()

    def reset_marker_axes() -> None:
        try:
            editor_state.reset_selected_marker_axes_to_reference()
        except ValueError as error:
            report_error(error)
            return
        draw_markers()
        update_panel(
            "Reference-axis orientation restored. Preview only: no project files are changed."
        )

    def on_pick(point) -> None:
        if point is None:
            return
        candidate, warning = snap_candidate_from_displayable_mesh_pick(point, displayable_meshes)
        click_mode_before_pick = editor_state.click_mode
        try:
            marker = editor_state.handle_geometry_pick(candidate)
        except ValueError as error:
            report_error(error)
            return

        if click_mode_before_pick == "create_marker":
            clear_selected_feature_cue()
            draw_markers()
            if warning is not None:
                print(warning)
            print(f"Created preview marker: {marker.name} at {_format_vector(marker.origin)}")
            update_panel()
            return

        if click_mode_before_pick in {"pick_primary_vector", "pick_secondary_vector"}:
            clear_selected_feature_cue()
            draw_markers()
            print(f"Assigned axis feature: {feature_summary(candidate)}")
            update_panel()
            return

        draw_selected_feature(candidate)
        print(f"Selected axis feature: {feature_summary(candidate)}")
        update_panel()

    click_mode.currentIndexChanged.connect(set_click_mode)
    new_marker_button.clicked.connect(begin_new_marker)
    marker_list.currentItemChanged.connect(lambda current, _previous: select_list_marker(current))
    apply_properties_button.clicked.connect(apply_properties)
    primary_axis.currentTextChanged.connect(set_primary_axis)
    secondary_axis.currentTextChanged.connect(set_secondary_axis)
    primary_reference_axis.currentTextChanged.connect(set_primary_reference_axis)
    secondary_reference_axis.currentTextChanged.connect(set_secondary_reference_axis)
    primary_reference.toggled.connect(set_primary_reference_mode)
    primary_feature.toggled.connect(set_primary_mode)
    secondary_reference.toggled.connect(set_secondary_reference_mode)
    secondary_feature.toggled.connect(set_secondary_mode)
    primary_use_feature.clicked.connect(begin_pick_primary_feature)
    secondary_use_feature.clicked.connect(begin_pick_secondary_feature)
    primary_flip_direction.clicked.connect(flip_primary_direction)
    secondary_flip_direction.clicked.connect(flip_secondary_direction)
    reset_axes.clicked.connect(reset_marker_axes)

    picking_options = {
        "callback": on_pick,
        "show_message": False,
        "left_clicking": True,
        "show_point": False,
    }
    try:
        plotter.enable_surface_point_picking(**picking_options)
    except TypeError:
        picking_options.pop("show_point")
        plotter.enable_surface_point_picking(**picking_options)

    plotter.add_axes()
    update_panel()
    window.resize(1320, 860)
    window.show()
    if owns_application:
        app.exec()
