"""Center the pinned RLBench sampler's single button spawn region."""


def restrict_to_center(boundary, scale):
    if not 0.0 < scale <= 1.0:
        raise ValueError("Layout scale must be in (0, 1].")
    if scale == 1.0:
        return

    # The area weights still select the only region with probability one.
    (region,) = boundary._boundaries
    box = region._boundary_bbox
    x_margin = (box.max_x - box.min_x) * (1.0 - scale) / 2.0
    y_margin = (box.max_y - box.min_y) * (1.0 - scale) / 2.0
    region._boundary_bbox = type(box)(
        box.min_x + x_margin, box.max_x - x_margin,
        box.min_y + y_margin, box.max_y - y_margin,
        box.min_z, box.max_z,
    )
