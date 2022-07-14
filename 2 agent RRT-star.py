'''
MIT License
Copyright (c) 2019 Fanjin Zeng
This work is licensed under the terms of the MIT license, see <https://opensource.org/licenses/MIT>.
'''
# idx is node number of each node: idx ranges from (0,n), where n is max number of nodes(iterations)
import numpy as np
from random import random
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from collections import deque
from shapely.geometry import LineString

class Line():
    ''' Define line '''

    def __init__(self, p0, p1):
        self.p = np.array(p0)
        self.dirn = np.array(p1) - np.array(p0)
        self.dist = np.linalg.norm(self.dirn)
        self.dirn /= self.dist  # normalize

    def path(self, t):
        return self.p + t * self.dirn


def Intersection(line, center, radius):
    ''' Check line-sphere (circle) intersection '''
    a = 1 + np.dot(line.dirn, line.dirn)
    b = 2 * np.dot(line.dirn, line.p - center)
    c = np.dot(line.p - center, line.p - center) - radius * radius

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return False

    t1 = (-b + np.sqrt(discriminant)) / (2 * a)
    t2 = (-b - np.sqrt(discriminant)) / (2 * a)

    if (t1 < 0 and t2 < 0) or (t1 > line.dist and t2 > line.dist):
        return False

    return True


def distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))


def isInObstacle(vex, obstacles, radius):
    for obs in obstacles:
        if distance(obs, vex) < radius:
            return True
    return False


def isThruObstacle(line, obstacles, radius):
    for obs in obstacles:
        if Intersection(line, obs, radius):
            return True
    return False


def nearest0(G, vex, obstacles, radius):
    Nvex = None
    Nidx = None
    minDist = float("inf")

    for idx, v in enumerate(G.vertices0):
        # print(idx," ",v)
        line = Line(v, vex)
        if isThruObstacle(line, obstacles, radius):
            continue

        dist = distance(v, vex)
        if dist < minDist:
            minDist = dist
            Nidx = idx
            Nvex = v

    return Nvex, Nidx


def nearest1(G, vex, obstacles, radius):
    Nvex = None
    Nidx = None
    minDist = float("inf")

    for idx, v in enumerate(G.vertices1):
        # print(idx," ",v)
        line = Line(v, vex)
        if isThruObstacle(line, obstacles, radius):
            continue

        dist = distance(v, vex)
        if dist < minDist:
            minDist = dist
            Nidx = idx
            Nvex = v

    return Nvex, Nidx


def newVertex(randvex, nearvex, stepSize):
    dirn = np.array(randvex) - np.array(nearvex)
    length = np.linalg.norm(dirn)
    dirn = (dirn / length) * min(stepSize, length)

    newvex = (nearvex[0] + dirn[0], nearvex[1] + dirn[1])
    return newvex


def window(startpos, endpos):
    ''' Define seach window - 2 times of start to end rectangle'''
    width = endpos[0] - startpos[0]
    height = endpos[1] - startpos[1]
    winx = startpos[0] - (width / 2.)
    winy = startpos[1] - (height / 2.)
    return winx, winy, width, height


def isInWindow(pos, winx, winy, width, height):
    ''' Restrict new vertex insides search window'''
    if winx < pos[0] < winx + width and \
            winy < pos[1] < winy + height:
        return True
    else:
        return False


class Graph:
    ''' Define graph '''

    def __init__(self, startpos, endpos):
        self.startpos0 = startpos[0]
        self.endpos0 = endpos[0]
        self.startpos1 = startpos[1]
        self.endpos1 = endpos[1]

        self.vertices0 = [startpos[0]]
        self.edges0 = []
        self.success0 = False
        self.vertices1 = [startpos[1]]
        self.edges1 = []
        self.success1 = False

        self.vex2idx0 = {startpos[0]: 0}
        self.neighbors0 = {0: []}
        self.distances0 = {0: 0.}
        self.vex2idx1 = {startpos[1]: 0}
        self.neighbors1 = {0: []}
        self.distances1 = {0: 0.}

        self.sx0 = endpos[0][0] - startpos[0][0]
        self.sy0 = endpos[0][1] - startpos[0][1]
        self.sx1 = endpos[1][0] - startpos[1][0]
        self.sy1 = endpos[1][1] - startpos[1][1]

    def add_vex0(self, pos):
        try:
            idx = self.vex2idx0[pos]
        except:
            idx = len(self.vertices0)
            self.vertices0.append(pos)
            self.vex2idx0[pos] = idx
            self.neighbors0[idx] = []
        return idx

    def add_vex1(self, pos):
        try:
            idx = self.vex2idx1[pos]
        except:
            idx = len(self.vertices1)
            self.vertices1.append(pos)
            self.vex2idx1[pos] = idx
            self.neighbors1[idx] = []
        return idx

    def add_edge0(self, idx1, idx2, cost):
        self.edges0.append((idx1, idx2))
        self.neighbors0[idx1].append((idx2, cost))
        self.neighbors0[idx2].append((idx1, cost))

    def add_edge1(self, idx1, idx2, cost):
        self.edges1.append((idx1, idx2))
        self.neighbors1[idx1].append((idx2, cost))
        self.neighbors1[idx2].append((idx1, cost))

    def randomPosition0(self):
        rx0 = random()
        ry0 = random()
        posx0 = self.startpos0[0] - (self.sx0 / 2.) + rx0 * self.sx0 * 2.0
        posy0 = self.startpos0[1] - (self.sy0 / 2.) + ry0 * self.sy0 * 2.0
        return posx0, posy0

    def randomPosition1(self):
        rx1 = random()
        ry1 = random()

        posx1 = self.startpos1[0] - (self.sx1 / 2.) + rx1 * self.sx1 * 2.0
        posy1 = self.startpos1[1] - (self.sy1 / 2.) + ry1 * self.sy1 * 2.0
        return posx1, posy1


def RRT(startpos, endpos, obstacles, n_iter, radius, stepSize):
    ''' RRT algorithm '''
    G = Graph(startpos, endpos)

    for _ in range(n_iter):
        # print("new call")
        randvex0 = G.randomPosition0()
        randvex1 = G.randomPosition1()
        # print(randvex0," ", randvex1)
        if isInObstacle(randvex0, obstacles, radius) or isInObstacle(randvex1, obstacles, radius):
            continue

        nearvex0, nearidx0 = nearest0(G, randvex0, obstacles, radius)
        nearvex1, nearidx1 = nearest1(G, randvex1, obstacles, radius)
        # print(nearvex0, " ", nearidx0,"   ",nearvex1," ",nearidx1)
        if nearvex0 is None or nearvex1 is None:
            continue

        newvex0 = newVertex(randvex0, nearvex0, stepSize)
        newvex1 = newVertex(randvex1, nearvex1, stepSize)

        newidx0 = G.add_vex0(newvex0)
        dist0 = distance(newvex0, nearvex0)
        G.add_edge0(newidx0, nearidx0, dist0)
        # print(newidx0, " ", dist0)

        newidx1 = G.add_vex1(newvex1)
        dist1 = distance(newvex1, nearvex1)
        G.add_edge1(newidx1, nearidx1, dist1)
        # print(newidx1, " ", dist1)

        dist2 = distance(newvex0, G.endpos0)
        dist3 = distance(newvex1, G.endpos1)
        if dist2 < stepSize:
            endidx0 = G.add_vex0(G.endpos0)
            G.add_edge0(newidx0, endidx0, dist0)
            G.success0 = True
        if dist3 < stepSize:
            endidx1 = G.add_vex1(G.endpos1)
            G.add_edge1(newidx1, endidx1, dist1)
            G.success1 = True

        '''
        print(newvex[0]," ",newvex[1])
        if abs(newvex[0]-endpos[0])<0.5 and abs(newvex[1]-endpos[1])<0.5:
            endidx = G.add_vex(G.endpos)
            G.add_edge(newidx, endidx, dist)
            G.success = True
            # print('success')
            # break
        '''
    return G


def RRT_star(startpos, endpos, obstacles, n_iter, radius, stepSize):
    ''' RRT star algorithm '''
    G = Graph(startpos, endpos)

    for _ in range(n_iter):
        randvex0 = G.randomPosition0()
        if isInObstacle(randvex0, obstacles, radius):
            continue

        nearvex0, nearidx0 = nearest0(G, randvex0, obstacles, radius)  # calculating nearest vertex in graph from randvex
        if nearvex0 is None:
            continue

        newvex0 = newVertex(randvex0, nearvex0, stepSize)  # create new vertex at that random vertex if conditions satisfied

        newidx0 = G.add_vex0(newvex0)
        dist0 = distance(newvex0, nearvex0)
        G.add_edge0(newidx0, nearidx0, dist0)
        G.distances0[newidx0] = G.distances0[nearidx0] + dist0

        # update nearby vertices distance (if shorter)
        for vex in G.vertices0:
            if vex == newvex0:  # newvex= newly generatex vertex
                continue

            dist1 = distance(vex, newvex0)
            if dist1 > radius:
                continue

            line = Line(vex, newvex0)
            if isThruObstacle(line, obstacles, radius):
                continue

            idx0 = G.vex2idx0[vex]
            if G.distances0[newidx0] + dist1 < G.distances0[idx0]:
                G.add_edge0(idx0, newidx0, dist1)
                G.distances0[idx0] = G.distances0[newidx0] + dist1

        dist2 = distance(newvex0, G.endpos0)
        if dist2 < stepSize:
            endidx0 = G.add_vex0(G.endpos0)
            G.add_edge0(newidx0, endidx0, dist2)
            try:
                G.distances0[endidx0] = min(G.distances0[endidx0], G.distances0[newidx0] + dist2)
            except:
                G.distances0[endidx0] = G.distances0[newidx0] + dist2

            G.success = True
            # print('success')
            # break

    for _ in range(n_iter):
        randvex1 = G.randomPosition1()
        if isInObstacle(randvex1, obstacles, radius):
            continue

        nearvex1, nearidx1 = nearest1(G, randvex1, obstacles, radius)       #calculating nearest vertex in graph from randvex
        if nearvex1 is None:
            continue

        newvex1 = newVertex(randvex1, nearvex1, stepSize)          #create new vertex at that random vertex if conditions satisfied

        newidx1 = G.add_vex1(newvex1)
        dist3 = distance(newvex1, nearvex1)
        G.add_edge1(newidx1, nearidx1, dist3)
        G.distances1[newidx1] = G.distances1[nearidx1] + dist3

        # update nearby vertices distance (if shorter)
        for vex in G.vertices1:
            if vex == newvex1:           #newvex= newly generatex vertex
                continue

            dist4 = distance(vex, newvex1)
            if dist4 > radius:
                continue

            line = Line(vex, newvex1)
            if isThruObstacle(line, obstacles, radius):
                continue

            idx = G.vex2idx1[vex]
            if G.distances1[newidx1] + dist4 < G.distances1[idx]:
                G.add_edge1(idx, newidx1, dist4)
                G.distances1[idx] = G.distances1[newidx1] + dist4

        dist5 = distance(newvex1, G.endpos1)
        if dist5 < stepSize:
            endidx1 = G.add_vex1(G.endpos1)
            G.add_edge1(newidx1, endidx1, dist5)
            try:
                G.distances1[endidx1] = min(G.distances1[endidx1], G.distances1[newidx1]+dist5)
            except:
                G.distances1[endidx1] = G.distances1[newidx1]+dist5

            G.success = True
            #print('success')
            # break
    return G


def dijkstra(G):
    '''
    Dijkstra algorithm for finding shortest path from start position to end.
    '''
    srcIdx0 = G.vex2idx0[G.startpos0]
    dstIdx0 = G.vex2idx0[G.endpos0]
    srcIdx1 = G.vex2idx1[G.startpos1]
    dstIdx1 = G.vex2idx1[G.endpos1]

    # build dijkstra
    nodes = list(G.neighbors0.keys())  # numbered nodes
    dist = {node: float('inf') for node in nodes}
    # print(dist)
    prev = {node: None for node in nodes}
    # print(prev)
    dist[srcIdx0] = 0

    while nodes:
        # print("new loop")
        curNode = min(nodes, key=lambda node: dist[node])
        # print(dist," ",curNode)
        nodes.remove(curNode)
        # print(curNode)
        if dist[curNode] == float('inf'):
            break
        for neighbor, cost in G.neighbors0[curNode]:
            # print(curNode," ",neighbor," ",cost," ",dist[curNode])
            newCost = dist[curNode] + cost
            # print(newCost," ",dist[neighbor])
            if newCost <= dist[neighbor]:
                dist[neighbor] = newCost
                prev[neighbor] = curNode

    # retrieve path
    path0 = deque()
    curNode = dstIdx0
    # print(curNode," ",dstIdx)
    while prev[curNode] is not None:
        # print("new loop")
        # print(prev[curNode]," ",curNode)
        path0.appendleft(G.vertices0[curNode])
        curNode = prev[curNode]
    path0.appendleft(G.vertices0[curNode])

    # build dijkstra
    nodes = list(G.neighbors1.keys())  # numbered nodes
    dist = {node: float('inf') for node in nodes}
    # print(dist)
    prev = {node: None for node in nodes}
    # print(prev)
    dist[srcIdx1] = 0

    while nodes:
        # print("new loop")
        curNode = min(nodes, key=lambda node: dist[node])
        # print(dist," ",curNode)
        nodes.remove(curNode)
        # print(curNode)
        if dist[curNode] == float('inf'):
            break
        for neighbor, cost in G.neighbors1[curNode]:
            # print(curNode," ",neighbor," ",cost," ",dist[curNode])
            newCost = dist[curNode] + cost
            # print(newCost," ",dist[neighbor])
            if newCost <= dist[neighbor]:
                dist[neighbor] = newCost
                prev[neighbor] = curNode

    # retrieve path
    path1 = deque()
    curNode = dstIdx1
    # print(curNode," ",dstIdx)
    while prev[curNode] is not None:
        # print("new loop")
        # print(prev[curNode]," ",curNode)
        path1.appendleft(G.vertices1[curNode])
        curNode = prev[curNode]
    path1.appendleft(G.vertices1[curNode])

    print(dstIdx0, " ", dstIdx1)

    return list(path0), list(path1)


def plot(G, obstacles, radius, path0=None, path1=None):
    '''
    Plot RRT, obstacles and shortest path
    '''
    px0 = [x for x, y in G.vertices0]
    py0 = [y for x, y in G.vertices0]
    px1 = [x for x, y in G.vertices1]
    py1 = [y for x, y in G.vertices1]
    fig, ax = plt.subplots()

    for obs in obstacles:
        circle = plt.Circle(obs, radius, color='red')
        ax.add_artist(circle)

    ax.scatter(px0, py0, c='cyan')
    ax.scatter(px1, py1, c='brown')
    ax.scatter(G.startpos0[0], G.startpos0[1], c='black')
    ax.scatter(G.endpos0[0], G.endpos0[1], c='black')
    ax.scatter(G.startpos1[0], G.startpos1[1], c='black')
    ax.scatter(G.endpos1[0], G.endpos1[1], c='black')

    lines = [(G.vertices0[edge[0]], G.vertices0[edge[1]]) for edge in G.edges0]
    lc0 = mc.LineCollection(lines, colors='green', linewidths=2)
    ax.add_collection(lc0)

    lines = [(G.vertices1[edge[0]], G.vertices1[edge[1]]) for edge in G.edges1]
    lc1 = mc.LineCollection(lines, colors='violet', linewidths=2)
    ax.add_collection(lc1)

    if path0 is not None:
        paths = [(path0[i], path0[i + 1]) for i in range(len(path0) - 1)]
        lc2 = mc.LineCollection(paths, colors='blue', linewidths=3)
        ax.add_collection(lc2)

    if path1 is not None:
        paths = [(path1[i], path1[i + 1]) for i in range(len(path1) - 1)]
        lc3 = mc.LineCollection(paths, colors='orange', linewidths=3)
        ax.add_collection(lc3)

    ax.autoscale()
    ax.margins(0.1)
    # for x in range(len(G.vex2idx0)):
    # plt.annotate(str(x), xy=(G.vertices0[x][0], G.vertices0[x][1]))

    # for x in range(len(G.vex2idx1)):
    # plt.annotate(str(x), xy=(G.vertices1[x][0], G.vertices1[x][1]))

    # print("plotted")
    # for x in range()
    plt.show()


def pathSearch(startpos, endpos, obstacles, n_iter, radius, stepSize):
    G = RRT_star(startpos)

    # Initialisation of the tree, to pos, endpos, obstacles, n_iter, radius, stepSize)
    if G.success:
        path = dijkstra(G)
        # plot(G, obstacles, radius, path)
        return path


if __name__ == '__main__':
    # for x in range(2):
    while True:
        startpos = [(0., 0.), (0., 5.)]
        endpos = [(5., 5.), (5., 0.)]
        obstacles = [(4., 4.), (2., 2.)]
        n_iter = 200
        radius = 0.5
        stepSize = 0.7
        check = 0
        numIntersection = 0
        # print(x)
        G = RRT_star(startpos, endpos, obstacles, n_iter, radius, stepSize)
        #G = RRT(startpos, endpos, obstacles, n_iter, radius, stepSize)
        #print(G.success0, " ", G.success1)
        # if G.success0 and G.success1:
        path0, path1 = dijkstra(G)
        #check = np.zeros((len(path0), len(path1)), dtype=bool)
        for x in range(len(path0) - 1):
            line = LineString([path0[x], path0[x + 1]])
            for y in range(len(path1) - 1):
                other = LineString([path1[y], path1[y + 1]])
                # check[x][y] = False
                if line.intersects(other):
                    if x == y:
                        print("Not valid", " ", x, " ", y)
                        numIntersection = numIntersection+1
                        check = 1
                    else:
                        print("Valid", " ", x, " ", y)
                        check = 0
                    # check[x][y] = True
        if check:
            continue
        print(path0)
        print(path1)
        plot(G, obstacles, radius, path0, path1)
        break
    # plot(G, obstacles, radius, 1, 1)
    # plt.show()
    # else:
    # plot(G, obstacles, radius)
