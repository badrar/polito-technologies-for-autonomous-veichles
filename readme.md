# Technologies for Autonomous Vehicles

This repository collects the two assignments developed for the *Technologies for Autonomous Vehicles* course (01SQHOV) at Politecnico di Torino. The goal of both was to build the core components of an autonomous driving stack from scratch — path planning on one side, perception on the other — using only standard libraries and explicit geometry, with no end-to-end deep learning shortcuts on the parts that didn't require them.

## Ass1 — Path planning on a real road network

The task was to run Dijkstra and A\* on the road network of a real city, downloaded from OpenStreetMap via OSMnx, and compare their behavior on realistic instances. Two cities were used: Turin and Aosta. Edge weights represent travel time, computed as edge length divided by the road's speed limit.

For A\*, three heuristics were implemented and compared: Euclidean, Manhattan, and Haversine. A non-trivial part of the work was ensuring admissibility with respect to the time-based cost. Since the graph coordinates are in geographic units, the heuristics require appropriate scale factors — roughly 111,320 / v_max for degree-based distances and 1,000 / v_max for Haversine in km — to remain a lower bound on the actual travel time. Without them, A\* silently loses optimality, which is only detectable by comparing its final costs against Dijkstra's.

One practical note: the OSMnx cache across library versions can yield significantly different graphs for the same query (in this case, 76k vs 11k nodes for Turin). If results look inconsistent across runs, the cache is the first thing to check.

## Ass2 — Lane detection with the GOLD algorithm

This assignment implements the GOLD algorithm (Bertozzi & Broggi) for forward-camera lane detection, applied to sequence 044 of the PandaSet dataset. The pipeline is written in Python using OpenCV and NumPy; no neural networks are used for lane detection itself.

The pipeline consists of:

- Analytical IPM/BEV transform derived from the PandaSet camera intrinsics (fx=fy≈1970, cx≈970, cy≈483, camera height ~1.66 m). The homography is computed from geometry rather than empirical calibration.
- GOLD-style horizontal gradient filter to enhance bright lane markings against a darker road surface.
- Geodesic dilation to reconnect dashed markers without saturating the rest of the image.
- Adaptive thresholding on local maxima, followed by connected-component filtering by aspect ratio.
- Per-lane polynomial fitting with temporal smoothing across consecutive frames.
- Solid/dashed classification based on the distribution of segments along the fitted curve.

An additional component, beyond the scope of the original paper (which assumed a stereo setup), is a monocular distance estimation for objects detected with YOLO. The bottom-center of each bounding box is projected onto the road plane via the inverse IPM transform. Estimates are reasonable within ~15 m; beyond that, a systematic overestimation of approximately 6 m emerges, which is the expected limitation of monocular distance recovery without real depth information. The behavior is documented in the report rather than hidden.

Two issues worth recording for future reference:

- The temporal filter contained a feedback loop that latched onto stale history values when a lane was missing for a few consecutive frames.
- A variable named `binary` retained its name after a Gaussian blur turned it into a grayscale image, which made the subsequent thresholding step harder to debug than it should have been.

## Running the code

Each assignment is self-contained in its own folder. The pipelines require Python 3.10+ with `opencv-python`, `numpy`, `networkx`, `osmnx`, and `ultralytics` (for the YOLO component). PandaSet is not included in the repository due to its size; dataset paths are configurable at the top of each script.

## Notes

This is academic work, not production code. Issues and suggestions are welcome.
