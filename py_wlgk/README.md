# Weisfeiler-Lehman Graph Kernel (WLGK) Unlabeled Implementation

This module implements the unlabeled version of the Weisfeiler-Lehman graph kernel (WLGK) which can
be used to compute a similarity score between two graphs. It works for both directed and undirected
graphs, and the node labels are ignored during computation.

This implementation was written in 1 hour so that I could understand the algorithm from first
principles. Don't use this in production, unless you want to improve it first. That being said, I
believe this implementation is correct and much faster than the other implementations I have seen,
which you can find below.

Other implementations:
- [WWL](https://github.com/BorgwardtLab/WWL)
- [jstsp2015](https://github.com/emanuele/jstsp2015/blob/master/gk_weisfeiler_lehman.py)


## Dependencies

- `networkx`: For graph representations and operations.
- `numpy`: For vector operations.

## How To Use

`wlgk(G1, G2, num_iterations=5) -> float`

- Parameters:
    - `G1`: First graph (`networkx.Graph` object).
    - `G2`: Second graph (`networkx.Graph` object).
    - `num_iterations`: Number of iterations for the WL process (default is 5).
- Returns:
    - A similarity score between the two graphs.

### Example

```python
import networkx as nx
from wlgk import wlgk

# Creating two example graphs
G1 = nx.Graph()
G1.add_edges_from([(1, 2), (2, 3)])

G2 = nx.Graph()
G2.add_edges_from([(1, 2), (2, 3), (3, 4)])

# Compute the WLGK similarity score
score = wlgk(G1, G2)
print(score)
```

### Implementation Details

The module uses a hashing mechanism to colour the nodes of the graph at each iteration of the Weisfeiler-Lehman process. The final node colourings are used to compute histograms for each graph which are then compared to compute the final similarity score.

