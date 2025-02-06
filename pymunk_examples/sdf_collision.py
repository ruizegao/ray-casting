import pygame
import sys
import pymunk as pm
import pymunk.pygame_util
from pymunk.vec2d import Vec2d

class SDFShape:
    def __init__(self, body, sdf_func, color=(0, 0, 255)):
        self.shape = pm.Circle(body, 1)  # Use a placeholder pymunk shape for collision space
        # Replace above "1" with a small radius or other parameter that fits your needs
        self.sdf_func = sdf_func  # Your signed distance function
        self.body = body  # Reference to the body
        self.color = color

    def point_query(self, point):
        """Override PyMunk's point query to use SDF"""
        distance = self.sdf_func(point)  # Signed distance function
        if distance <= 0:
            return pm.PointQueryInfo(self.shape, point, -distance, Vec2d(0, 0))  # Inside collision
        return pm.PointQueryInfo(None, point, distance, Vec2d(0, 0))  # No collision

    def segment_query(self, a, b):
        """Implement segment queries if needed for raycasting"""
        return None  # Implement if you want ray intersections

def sdf_circle(point, center=Vec2d(100, 100), radius=50):
    return (point - center).length - radius  # Positive: outside, Negative: inside

def collision_handler(sdf_shape, mouse_body):
    """Custom collision checking between the SDF object and the mouse"""
    mouse_position = Vec2d(*mouse_body.position)
    sdf_value = sdf_shape.sdf_func(mouse_position)
    return sdf_value <= 0  # True if the mouse is colliding with the SDF object


def draw_sdf_circle(screen, sdf_shape, resolution=400):
    """Manually draw the SDF shape as a circle"""
    center = (int(sdf_shape.body.position.x), int(sdf_shape.body.position.y))
    radius = 10  # Hardcoded for visualization purposes; todo: dynamically adjust

    for x in range(-radius, radius):
        for y in range(-radius, radius):
            point = Vec2d(center[0] + x, center[1] + y)
            sdf_value = sdf_shape.sdf_func(point)

            if sdf_value <= 0:  # Inside the shape
                color = sdf_shape.color
                screen.set_at((int(point.x), int(point.y)), color)


def main():

    # initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 20)

    # create a space and tell PyMunk to draw to the above screen
    space = pm.Space()
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    body = pm.Body(body_type=pm.Body.STATIC)  # Static body for the shape
    body.position = 600, 200
    sdf_obj = SDFShape(body, lambda p: sdf_circle(p, center=Vec2d(400, 300), radius=50))

    space.add(body, sdf_obj.shape)

    handler = space.add_default_collision_handler()
    handler.begin = collision_handler

    # Mouse Circle (as a dynamic object)
    mouse_body = pm.Body(body_type=pm.Body.KINEMATIC)  # Mouse-controlled body
    mouse_radius = 20
    mouse_shape = pm.Circle(mouse_body, mouse_radius)  # Circle shape for the mouse
    mouse_shape.color = pygame.Color("red")
    space.add(mouse_body, mouse_shape)


    while True:
        for event in pygame.event.get():
            if (
                event.type == pygame.QUIT
                or event.type == pygame.KEYDOWN
                and event.key == pygame.K_ESCAPE
            ):
                return
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                pygame.image.save(screen, "sdf_collisions.png")

        p = pygame.mouse.get_pos()
        mouse_body.position = p[0], p[1]

        # Collision logic
        show_collision = collision_handler(sdf_obj, mouse_body)

        screen.fill(pygame.Color("white"))
        space.debug_draw(draw_options)

        # Draw the SDF object manually
        draw_sdf_circle(screen, sdf_obj)

        # Draw the mouse object (red if no collision, green if collision)
        mouse_color = pygame.Color("green") if show_collision else pygame.Color("red")
        pygame.draw.circle(screen, mouse_color, (int(mouse_body.position.x), int(mouse_body.position.y)), mouse_radius)

        screen.blit(
            font.render(
                "Left click to switch shape type, right click to rotate. (The shape follows the mouse)",
                True,
                pygame.Color("black"),
            ),
            (5, 5),
        )

        space.step(1.0 / 60.0)

        pygame.display.flip()
        clock.tick(50)

if __name__ == "__main__":
    sys.exit(main())