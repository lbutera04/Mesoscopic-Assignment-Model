# Mesoscopic Assignment Model

A scalable framework for **mesoscopic traffic assignment and route sampling** on large road networks, designed for urban-scale freight and passenger simulations.

This project implements a **corridor-aware path sampling model** that generates realistic route diversity while remaining computationally tractable on large networks. The system builds directed acyclic corridor structures from road networks, samples feasible paths efficiently, and exports routes for simulation platforms such as **SUMO**.

The model is designed to support **large-scale transportation simulations** such as freight corridor studies, fast traffic simulation, and digital-twin transportation modeling.

---

# Overview

Traditional traffic assignment methods struggle to scale to very large networks while maintaining realistic route diversity. This project explores an alternative strategy:

1. Construct a **corridor-restricted directed acyclic graph (DAG)** between origin–destination pairs.
2. Efficiently sample feasible routes within this structure.
3. Generate route sets suitable for mesosimulation.

The system emphasizes:

* scalability to large OSM networks
* realistic route diversity
* compatibility with simulation environments
* efficient sampling using compiled kernels

The project was developed as part of research on **corridor-scale freight modeling and transportation digital twins**.

---

# Key Features

### Corridor DAG Construction

Builds directed acyclic subgraphs representing feasible travel corridors between OD pairs.

Features:

* reachability pruning
* corridor width control
* structural diagnostics
* two-tree corridor expansion

Core modules:

```
cccar/corridor/
    core.py
    dag.py
    twotree_web.py
```

---

### Efficient Path Sampling

Routes are sampled from the corridor DAG using high-performance kernels implemented with:

* **Numba**
* **C++ / PyBind**

Sampling supports:

* weighted path sampling
* large OD sets
* reproducible route generation

Core modules:

```
cccar/sampling/
    api.py
    api_twotree_web.py
    numba_kernels.py
    python_impl.py
```

C++ kernels:

```
cpp/
    sampler.cpp
    corridor_core.cpp
```

---

### Road Network Construction from OSM

The system converts OpenStreetMap data into a graph suitable for assignment and simulation.

Capabilities:

* edge attribute extraction
* CSR graph construction
* centroid attachment
* spatial utilities

Core modules:

```
cccar/osm/
    graph_build.py
    attributes.py
    geo.py
```

---

### Demand Generation

Demand can be generated from multiple sources, including **Replica OD data**.

Modules:

```
cccar/demand/
    replica.py
    centroids.py
    spawns.py
```

---

### SUMO Route Generation

Sampled paths can be exported into formats compatible with **SUMO mesosimulation**.

Modules:

```
cccar/routes/
    build.py
    sumo_io.py
```

---

### Evaluation and Diagnostics

Tools are included to analyze the structural properties of corridor DAGs and compare simulated flows with observed distributions.

Modules:

```
cccar/eval/
    link_volumes.py
    distribution_compare.py
```

Diagnostics scripts:

```
cccar/tools/
    dag_diagnostics.py
    dag_core_structure_diagnostics.py
    dag_benchmarks.py
```

These tools measure metrics such as:

* DAG density
* dominator structure
* bridge counts
* path diversity
* structural redundancy

---

# Architecture

The typical workflow is:

```
OSM Network
      │
      ▼
Graph Construction
(cccar.osm)
      │
      ▼
Demand Generation
(cccar.demand)
      │
      ▼
Corridor DAG Construction
(cccar.corridor)
      │
      ▼
Route Sampling
(cccar.sampling)
      │
      ▼
Route Export
(cccar.routes)
      │
      ▼
Simulation / Evaluation
```

---

# Installation

Clone the repository:

```
git clone https://github.com/lbutera04/Mesoscopic-Assignment-Model.git
cd Mesoscopic-Assignment-Model
```

Install dependencies:

```
pip install -r requirements.txt
```

Install the package:

```
pip install -e .
```

Optional: build the C++ extensions

```
mkdir build
cd build
cmake ..
make
```

---

# Basic Usage

Example workflow:

1. Build a road network from OSM
2. Generate OD demand
3. Construct corridor DAGs
4. Sample routes
5. Export to SUMO

Example CLI usage:

```
python -m cccar.cli ...
```

(Full examples are currently under development.)

---

# Project Structure

```
Mesoscopic-Assignment-Model
│
├── src/cccar
│   ├── corridor      corridor DAG construction
│   ├── sampling      route sampling algorithms
│   ├── demand        OD generation and processing
│   ├── osm           OSM network construction
│   ├── routes        route export utilities
│   ├── eval          evaluation metrics
│   └── tools         diagnostics and benchmarking
│
├── cpp               high-performance sampling kernels
├── cache             cached intermediate data
└── requirements.txt
```

---

# Research Motivation

Transportation simulations often require large route sets to represent realistic traveler behavior. Traditional shortest-path assignment tends to underestimate route diversity and is computationally expensive at large scales.

This project explores **corridor-based route sampling**, which:

* restricts routing to feasible geographic corridors
* maintains realistic path diversity
* scales to large networks
* supports mesosimulation environments

The system is particularly useful for applications such as:

* freight corridor analysis
* truck parking studies
* infrastructure resilience modeling
* accessibility analysis
* transportation digital twins

---

# Future Work

Planned future works include:

* Machine code implementations of graph algorithms
* Parallelization of preprocessing for increased speed
* Generalizability of data loading via GeoFabrik APIs
* Compilation of algorithm into realtime mesosimulator via SUMO API
* Custom mesosimulation model for maximum compatibility
* Realtime Traffic Analysis workflow for scenario testing
* UX/UI integration

---

# License

MIT License

---

# Author

Luca Butera
UC Berkeley — Applied Mathematics, Data Science, Economics
Transportation Data Science & Network Optimization
