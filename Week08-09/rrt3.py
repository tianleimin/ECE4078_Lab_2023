import math
import numpy as np

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

class RrtConnect:
    def __init__(self, s_start, s_goal, step_len, goal_sample_rate, iter_max, external_obstacles):
        self.s_start = Node(*s_start)
        self.s_goal = Node(*s_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.iter_max = iter_max
        self.V1 = [self.s_start]
        self.V2 = [self.s_goal]

        self.external_obstacles = external_obstacles  # External list of obstacles

    def planning(self, goal_tolerance):
        for i in range(self.iter_max):
            node_rand = self.generate_random_node()
            node_near = self.nearest_neighbor(self.V1, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if node_new and not self.is_collision(node_near, node_new):
                self.V1.append(node_new)
                node_near_prim = self.nearest_neighbor(self.V2, node_new)
                node_new_prim = self.new_state(node_near_prim, node_new)

                if node_new_prim and not self.is_collision(node_new_prim, node_near_prim):
                    self.V2.append(node_new_prim)

                    while True:
                        node_new_prim2 = self.new_state(node_new_prim, node_new)
                        if node_new_prim2 and not self.is_collision(node_new_prim2, node_new_prim):
                            self.V2.append(node_new_prim2)
                            node_new_prim = self.change_node(node_new_prim, node_new_prim2)
                        else:
                            break

                        if self.is_node_same(node_new_prim, node_new):
                            break

                    if self.is_node_within_goal(node_new_prim, goal_tolerance):
                        return self.extract_path(node_new, node_new_prim)

                if self.is_node_same(node_new_prim, node_new):
                    return self.extract_path(node_new, node_new_prim)

            if len(self.V2) < len(self.V1):
                list_mid = self.V2
                self.V2 = self.V1
                self.V1 = list_mid

        return None

    @staticmethod
    def change_node(node_new_prim, node_new_prim2):
        node_new = Node(node_new_prim2.x, node_new_prim2.y)
        node_new.parent = node_new_prim
        return node_new

    @staticmethod
    def is_node_same(node_new_prim, node_new):
        return node_new_prim.x == node_new.x and node_new_prim.y == node_new.y

    def generate_random_node(self):
        delta = self.step_len

        if np.random.random() > self.goal_sample_rate:
            return Node(
                np.random.uniform(self.s_start.x - delta, self.s_start.x + delta),
                np.random.uniform(self.s_start.y - delta, self.s_start.y + delta)
            )

        return self.s_goal

    @staticmethod
    def nearest_neighbor(node_list, n):
        return min(node_list, key=lambda nd: math.hypot(nd.x - n.x, nd.y - n.y))

    def new_state(self, node_start, node_end):
        dist, theta = self.get_distance_and_angle(node_start, node_end)

        dist = min(self.step_len, dist)
        x_new = node_start.x + dist * math.cos(theta)
        y_new = node_start.y + dist * math.sin(theta)
        node_new = Node(x_new, y_new)
        node_new.parent = node_start

        return node_new

    @staticmethod
    def extract_path(node_new, node_new_prim):
        path1 = [(node_new.x, node_new.y)]
        node_now = node_new

        while node_now.parent is not None:
            node_now = node_now.parent
            path1.append((node_now.x, node_now.y))

        path2 = [(node_new_prim.x, node_new_prim.y)]
        node_now = node_new_prim

        while node_now.parent is not None:
            node_now = node_now.parent
            path2.append((node_now.x, node_now.y))

        return list(reversed(path1)) + path2

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    def is_collision(self, node_start, node_end, safety_distance=0.0):
        num_samples = 10  # Number of samples between start and end for collision checking

        for i in range(num_samples + 1):
            x_sample = node_start.x + (i / num_samples) * (node_end.x - node_start.x)
            y_sample = node_start.y + (i / num_samples) * (node_end.y - node_start.y)

            # Check if the sampled point is within a certain distance of external obstacles
            for (ox, oy) in self.external_obstacles:
                if math.hypot(x_sample - ox, y_sample - oy) <= safety_distance:
                    return True  # Collision with an external obstacle

        return False  # No collision detected

    def is_node_within_goal(self, node, goal_tolerance):
        dist_to_goal = math.hypot(node.x - self.s_goal.x, node.y - self.s_goal.y)
        return dist_to_goal <= goal_tolerance

def main():
    # Define your external obstacles as (x, y) coordinates
    external_obstacles = [(1.0, 1.0), (-1.0, -1.0), (0.5, 0.5)]  # Example obstacles

    # Define your start and goal positions
    x_start = (0, 0)  # Starting node at the center
    x_goal = (2, 2)   # Goal node

    # Create an instance of RrtConnect with the updated parameters
    rrt_conn = RrtConnect(x_start, x_goal, 0.1, 0.05, 1000, external_obstacles)

    # Specify the goal tolerance (e.g., 0.1 meters)
    goal_tolerance = 0.1

    # Perform path planning with goal tolerance
    path = rrt_conn.planning(goal_tolerance)

    # Check if a valid path was found
    if path:
        print("Path found:", path)
        # Visualize or use the path as needed
    else:
        print("No valid path found.")

if __name__ == '__main__':
    main()
