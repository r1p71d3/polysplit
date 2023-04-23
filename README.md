# polysplit

A lightweight library for splitting polygons into regions based on proximity of points.\
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![issues](https://img.shields.io/github/issues/r1p71d3/polysplit)
[![codecov](https://codecov.io/gh/r1p71d3/polysplit/branch/main/graph/badge.svg?token=8S2VJLZG7U)](https://codecov.io/gh/r1p71d3/polysplit)
[![Build Status](https://github.com/r1p71d3/polysplit/actions/workflows/build.yml/badge.svg)](https://github.com/r1p71d3/polysplit/actions/workflows/build.yml)


## Overview
Map quantization is the procedure of dividing a continuous map into a number of discrete regions. The simplest approach that has been used for hundreds of years is to overlap the map with a square grid. However, this approach ignores the geographical features of the map, making it suboptimal for certain applications. With this project, I would like to propose a novel algorithm that organically divides any given map into regions based on the relative travel time between different areas.

In its simple form, the proposed algorithm could be applied to a map (a polygon) with intraversible obstacles (holes). It works in 2 stages. In the first stage, the map is overlayed with a fine grid of $N$ points. Then, we calculate the shortest path (around the obstacles) betweeen every pair of points and construct the $N \times N$ distance matrix. In the second stage, we apply the k-medoids algorithm to the set of points, using the matrix from stage I as a distance function, and retrieve a set of $M$ centers. Finally, we construct a Voronoi graph around the centers, creating the regions.

## Installation

`pip install polysplit`

## Sample usage

```python
import polysplit
from shapely.geometry import Polygon

# Create a polygon with a hole
outer_coords = [(0, 0), (0, 1), (1, 1), (1, 0)]
hole_coords = [(0.4, 0.4), (0.4, 0.6), (0.6, 0.6), (0.6, 0.4)]
polygon = Polygon(outer_coords, [hole_coords])

# Split the polygon
regions = polysplit.polysplit_main(polygon, k=5, num_points=100, plot=True)
print(regions)
```
