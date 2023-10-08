import math

class Utils:
    def __init__(self, obstacles, safety_distance=0.5):
        self.safety_distance = safety_distance
        self.obstacles = obstacles

    def is_collision(self, node_start, node_end):
        for obstacle in self.obstacles:
            distance = math.hypot(node_start.x - obstacle[0], node_start.y - obstacle[1])
            if distance < self.safety_distance:
                return True
        return False



    @staticmethod
    def get_ray(start, end):
        orig = [start.x, start.y]
        direc = [end.x - start.x, end.y - start.y]
        return orig, direc

    @staticmethod
    def get_dist(start, end):
        return math.hypot(end.x - start.x, end.y - start.y)