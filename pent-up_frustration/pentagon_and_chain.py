import numpy as np
from math import pi
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from itertools import combinations
from numba import njit
from typing import TypeVar
from collections.abc import Sequence
from numpy.typing import NDArray

TPentagon = TypeVar('TPentagon', bound='Pentagon')
TChain = TypeVar('TChain', bound='Chain')


class Pentagon:
    def __init__(self,
                 centroid: Sequence[int, int],
                 direction: -1 | 1,
                 busy_edge: int = None) -> None:
        '''
        creates a pentagon given a centroid and a direction
        
        Parameters:
            centroid:   x and y coordinates of the centroid
            direction:  1 means its top is pointy and bottom is flat, 
                        -1 is the opposite
            busy_edge:  the edge of 

        if direction == 1 edges are counted clockwise from the top 
        if direction == -1 edges are counted anticlockwise from the bottom 

        '''
        self.centroid = np.array(centroid)
        self.direction = direction
        h = 1 / 2 / np.tan(np.radians(180 / 5))
        self.r = ((1 / 2)**2 + h**2)**(1 / 2)
        self._get_vertices()
        self._get_edge_midpoints()
        self._get_edges()
        self.busy_edge = busy_edge

    def _get_vertices(self) -> None:

        c_1 = np.cos((2 * pi) / 5)
        c_2 = np.cos(pi / 5)
        s_1 = np.sin((2 * pi) / 5)
        s_2 = np.sin((4 * pi) / 5)

        vertices = np.array([(0, 1), (s_1, c_1), (s_2, -c_2), (-s_2, -c_2),
                             (-s_1, c_1)])
        vertices *= self.r

        if self.direction == -1:
            vertices *= np.array([1, -1])

        self.vertices = self.centroid + vertices

    def _get_edges(self) -> None:
        self.edges = np.hstack(
            [self.vertices,
             np.vstack([self.vertices[1:], self.vertices[0]])])

    def _get_edge_midpoints(self) -> None:
        mids = np.vstack([self.vertices, self.vertices[0]])
        self.edge_midpoints = (mids[1:] + mids[:-1]) / 2

    def adj_centroid_info(self) -> None:
        centroids = self.centroid + (self.edge_midpoints - self.centroid) * 2
        direction = self.direction * -1
        return centroids, direction

    def _overlap_type(self, other: TPentagon):
        return _overlap_type_inner(self.centroid, other.centroid, self.r)

    def overlaps(self, other: TPentagon) -> bool:
        '''
        Checks wither the pentagon passed overlaps with the current one
        
        Parameters:
            other (Pentagon instance): another instance of Pentagon to 
            compare to the calling one
        
        Returns 
            bool - True if overlaps, else False
        '''

        obv_overlap, may_overlap = self._overlap_type(other)
        if obv_overlap:
            return True
        if may_overlap:  #  check if any of the edges overlap
            n_overlaps = 0
            for v1 in range(5):
                for v2 in range(5):
                    n_overlaps += _do_intersect(self.edges[v1],
                                                other.edges[v2])
            if n_overlaps > 25:  # then it's chained
                return True
        return False

    def distance(self, other: TPentagon) -> float:
        '''
        Computes distance  between the calling Pentagon and another

        Paramters:
            other (Pentagon instance): another instance of Pentagon to 
            compare to the calling one
        
        Returns:
            float - distance between the two pentagons

        TODO 
            some comparison are superfluous
            dispatch to dist_lineline can be written better
        '''

        dist = _distance_inner(self.edges, other.edges)
        return dist


class Chain:
    def __init__(self, pentagons: list[TPentagon] = None) -> None:
        # TODO the option to initialize with pentagons can be turned into
        # an initializion from summary
        '''
        A chain of pentagons
        '''
        if pentagons is None:
            self.pentagons = [Pentagon((0, 0), 1)]
        else:
            self.pentagons = pentagons
        self.summary = tuple()
        self.error = None

    def from_summary(self, history: NDArray) -> tuple[bool, int]:
        # assert len(self.pentagons)==1, 'the chain is not empty'
        for i, x in enumerate(history):
            success = self.make_new(edge=x)
            if not success:
                return False, i
        return True, i

    def plot(self, path: str | None = None) -> None:
        '''
        Plots the chain of pentagons

        Parameters:
            path (str): If provided, saves the image to path
        '''
        chain_vertices = np.vstack([x.vertices for x in self.pentagons])
        ax = plt.gca()
        ax.set_aspect('equal')
        for n, x in enumerate(self.pentagons):
            ax.add_patch(Polygon(x.vertices))
            ax.text(*x.centroid,
                    str(n),
                    ma='center',
                    ha='center',
                    fontsize=14,
                    c='white')

        xmax, ymax = chain_vertices.max(axis=0)
        xmin, ymin = chain_vertices.min(axis=0)
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax)

        if path:
            plt.savefig(path)
            plt.close()
        else:
            plt.show()

    def _new_pentagon(self, p: TPentagon) -> bool:
        if not self.overlaps_existing(p):
            self.pentagons.append(p)
            self.summary += tuple([4 - p.busy_edge])
            return True
        return False

    def make_new(self,
                 edge: int | None = None,
                 free_edge: int | None = None,
                 random: bool = False) -> bool:
        '''
        Tries to add a pentagon to the end of the chain that does not overlap with the rest of it 
        Each arg corresponds to a different way of making a pentagon 

        Please note that edges are counted differently depending on how the pentagon is oriented:
        if direction == 1 edges are counted clockwise from the top, 
        while if direction == -1 edges are counted anticlockwise from the bottom 
        
        Parameters:
            edge (int): if provided, tries to make a pentagon that borders on 
            the edge stated

            free_endge (int): if provided, tries to make a pentagon that borders 
            on the edge stated - edges are counted ignoring any busy side - i.e. 
            if the first edge is busy - edge 0 will be the next available edge

            random (bool): if True, adds a pentagon on a random side
        Returns:
            bool - True if successfully made a pentagon that does not overlap with the chain,
            false otherwise
        '''
        p = self.pentagons[-1]
        c, s = p.adj_centroid_info()

        if edge is not None:
            p = Pentagon(c[edge], direction=s, busy_edge=4 - edge)
            success = self._new_pentagon(p)
            return success

        if free_edge is not None:
            possible_edges = [x for x in range(5) if x != p.busy_edge]
            edge = possible_edges[free_edge]
            p = Pentagon(c[edge], direction=s, busy_edge=4 - edge)
            success = self._new_pentagon(p)
            return success

        if random:
            possible_edges = [x for x in range(5) if x != p.busy_edge]
            np.random.shuffle(possible_edges)
            for edge in possible_edges:
                p = Pentagon(c[edge], direction=s, busy_edge=4 - edge)
                success = self._new_pentagon(p)
                if success:
                    return True
            return False

        raise ValueError('no edge provided')

    def min_nonzero_distance(self) -> None | int:
        '''
        Computes the minimum non-zero distance between any pentagon of the chain
        The distance between pentagon A and pentagon B is the infimum of the set 
        of distances between any point of A and any point of B 
        '''
        n_pentagons = len(self.pentagons)
        ps = self.pentagons
        out = {}
        for x, y in combinations(range(n_pentagons), 2):
            if not ps[x]._overlap_type(ps[y])[0]:
                dist = _dist_pointpoint(ps[x].centroid, ps[y].centroid)
                if dist < (3 * ps[x].r):
                    out[f'{x} {y}'] = ps[x].distance(ps[y])
        out = {k: v for k, v in out.items() if round(v, 12) > 0}

        if not out:
            return None

        return min(out)

    def overlaps_existing(self, pentagon: TPentagon) -> bool:
        '''checks wither the pentagon given overlaps with the existing chain'''
        relevant_chain = self.pentagons[:-1]
        for i, x in enumerate(relevant_chain):
            if pentagon.overlaps(x):
                self.error = {'pentagon_tried': pentagon, 'overlaps_with': i}
                return True
        return False


@njit
def _distance_inner(s1: NDArray[float], s2: NDArray[float]) -> float:
    dists = [
        _dist_lineline(s1[x, :2], s1[x, 2:4], s2[y, :2], s2[y, 2:4])
        for x in range(s1.shape[0]) for y in range(s2.shape[0])
    ]
    dist = min(dists)
    return dist


@njit
def _overlap_type_inner(centroid_1: Sequence[float, float],
                        centroid_2: Sequence[float, float],
                        r: float) -> tuple[bool, bool]:
    dist = _dist_pointpoint(centroid_1, centroid_2)
    dist = round(dist, 12)
    min_dist = np.cos(pi / 5) * 2 * r
    min_dist = round(min_dist, 12)
    obv_overlap = dist < min_dist
    may_overlap = dist < (2 * r)
    return obv_overlap, may_overlap


@njit
def _dist_pointpoint(x, y):
    return ((x - y)**2).sum()**(1 / 2)


@njit
def _onSegment(p, q, r):
    if ((q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0]))
            and (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):

        return True
    return False


@njit
def _orientation(p, q, r):
    tol = 1e-12
    val = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))
    if (val > tol):
        return 1
    elif (val < -tol):
        return 2
    else:
        return 0


@njit
def _do_intersect(s1, s2):

    args = np.hstack((s1, s2)).reshape((4, 2))

    p1, q1, p2, q2 = [(x, y) for x, y in args]

    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)

    # Special Cases
    if 0 in [o1, o2, o3, o4]:
        return 1

    # General case
    if ((o1 != o2) and (o3 != o4)):
        return 100

    # If none of the cases
    return 0


@njit
def _dist_linepoint(a, b, n):  # c is the point
    (x1, y1), (x2, y2), (x3, y3) = (a, b, n)

    px = x2 - x1
    py = y2 - y1

    norm = px * px + py * py

    u = ((x3 - x1) * px + (y3 - y1) * py) / float(norm)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    dist = (dx * dx + dy * dy)**.5
    return dist


@njit
def _dist_lineline(a, b, n, m):
    dists = [
        _dist_linepoint(a, b, n),
        _dist_linepoint(a, b, m),
        _dist_linepoint(n, m, a),
        _dist_linepoint(n, m, b)
    ]

    dist = min(dists)
    return dist