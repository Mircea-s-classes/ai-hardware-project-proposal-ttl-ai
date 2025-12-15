# Your current "happy" inference settings
THR = 0.22

POST = dict(
    min_dist_to_edge=18,
    min_area=40,
    max_area=12000,
    min_solidity=0.75,
    min_ellipticity=0.25,
    min_circularity=0.28,
    max_aspect_ratio=5.0,
    min_major_px=45,
    min_minor_px=25,
    min_ellipse_area=1300.0,
    close_iters=2,
    ellipse_refill=True,
)
