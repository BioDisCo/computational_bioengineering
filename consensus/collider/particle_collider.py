from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from enum import Enum
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pygame
from scipy import stats


type Color = tuple[int, int, int]
type Position = tuple[int, int]
type Reaction = dict[tuple[str, str], tuple[str | None, str | None]]


class BoundaryType(Enum):
    PERIODIC = "periodic"
    REFLECTIVE = "reflective"


class ParticleType(TypedDict):
    radius: int
    color: Color
    type: str


PARTICLE_TYPES: dict[str, ParticleType] = {
    "small": {"radius": 2, "color": (255, 100, 100), "type": "small"},
    "medium": {"radius": 4, "color": (100, 255, 100), "type": "medium"},
    "large": {"radius": 8, "color": (100, 100, 255), "type": "large"},
    "huge": {"radius": 16, "color": (100, 100, 100), "type": "huge"},
    #
    "A": {"radius": 16, "color": (76, 130, 183), "type": "A"},
    "B": {"radius": 16, "color": (239, 134, 54), "type": "B"},
}


class Particle:
    def __init__(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        radius: int,
        color: Color,
        particle_type: str,
    ) -> None:
        self.x: float = x
        self.y: float = y
        self.vx: float = vx
        self.vy: float = vy
        self.radius: int = radius
        self.color: Color = color
        self.type: str = particle_type
        self.mass: float = radius**2  # assume constant density

    def update(
        self,
        dt: float,
        width: int,
        height: int,
        boundary_type: BoundaryType = BoundaryType.PERIODIC,
    ) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt

        if boundary_type == BoundaryType.PERIODIC:
            # Periodic boundary conditions
            self.x = self.x % width
            self.y = self.y % height
        elif boundary_type == BoundaryType.REFLECTIVE:
            # Reflective boundary conditions
            if self.x < self.radius:
                self.x = self.radius
                self.vx = -self.vx
            elif self.x > width - self.radius:
                self.x = width - self.radius
                self.vx = -self.vx

            if self.y < self.radius:
                self.y = self.radius
                self.vy = -self.vy
            elif self.y > height - self.radius:
                self.y = height - self.radius
                self.vy = -self.vy


class SpatialGrid:
    def __init__(self, width: int, height: int, cell_size: float) -> None:
        self.width: int = width
        self.height: int = height
        self.cell_size: float = cell_size
        self.cols: int = int(width // cell_size) + 1
        self.rows: int = int(height // cell_size) + 1
        self.grid: defaultdict[tuple[int, int], list[Particle]] = defaultdict(list)

    def clear(self) -> None:
        self.grid.clear()

    def add_particle(self, particle: Particle) -> None:
        col = int(particle.x // self.cell_size)
        row = int(particle.y // self.cell_size)

        # Handle periodic boundaries
        col = col % self.cols
        row = row % self.rows

        self.grid[(col, row)].append(particle)

    def get_nearby_particles(self, particle: Particle) -> list[Particle]:
        col = int(particle.x // self.cell_size)
        row = int(particle.y // self.cell_size)

        nearby: list[Particle] = []
        for dc in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                check_col = (col + dc) % self.cols
                check_row = (row + dr) % self.rows
                nearby.extend(self.grid.get((check_col, check_row), []))

        return nearby


class ParticleCollider:
    def __init__(
        self,
        width: int = 400,
        height: int = 400,
        particle_counts: dict[str, int] | None = None,
        boundary_type: BoundaryType = BoundaryType.PERIODIC,
        reactions: Reaction | None = None,
        speed: float | None = None,
    ) -> None:
        pygame.init()
        self.width: int = width
        self.height: int = height
        self.boundary_type: BoundaryType = boundary_type
        self.screen: pygame.Surface = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Particle Collider")
        self.clock: pygame.time.Clock = pygame.time.Clock()

        # Handle particle count parameters
        if particle_counts is None:
            particle_counts = {"small": 400, "medium": 400, "large": 200}

        # Spatial grid for efficient collision detection
        max_radius: int = max(ptype["radius"] for ptype in PARTICLE_TYPES.values())
        self.grid: SpatialGrid = SpatialGrid(width, height, max_radius * 3)

        # Store speed parameter
        self.speed: float | None = speed

        self.particles: list[Particle] = self.create_particles_by_type(particle_counts)

        # Reaction system
        self.reactions: Reaction = reactions or {}

        # Collision tracking
        self.collision_times: dict[tuple[str, str], list[float]] = {}
        self.start_time: float = 0.0

        # Reaction tracking for CSV
        self.reaction_log: list[dict] = []

    def create_particles_by_type(
        self, particle_counts: dict[str, int]
    ) -> list[Particle]:
        particles: list[Particle] = []

        for type_name, count in particle_counts.items():
            if type_name not in PARTICLE_TYPES:
                raise ValueError(f"Unknown particle type: {type_name}")

            ptype: ParticleType = PARTICLE_TYPES[type_name]

            for _ in range(count):
                x: float = random.uniform(ptype["radius"], self.width - ptype["radius"])
                y: float = random.uniform(
                    ptype["radius"], self.height - ptype["radius"]
                )

                if self.speed is not None:
                    # Use fixed speed with random direction
                    angle = random.uniform(0, 2 * math.pi)
                    vx: float = self.speed * math.cos(angle)
                    vy: float = self.speed * math.sin(angle)
                else:
                    # Use random velocity components
                    vx: float = random.uniform(-50, 50)
                    vy: float = random.uniform(-50, 50)

                particle: Particle = Particle(
                    x, y, vx, vy, ptype["radius"], ptype["color"], ptype["type"]
                )
                particles.append(particle)

        return particles

    def distance_periodic(self, p1: Particle, p2: Particle) -> float:
        if self.boundary_type == BoundaryType.REFLECTIVE:
            # Simple Euclidean distance for reflective boundaries
            dx = p1.x - p2.x
            dy = p1.y - p2.y
            return math.sqrt(dx * dx + dy * dy)
        else:
            # Periodic distance calculation
            dx: float = abs(p1.x - p2.x)
            dy: float = abs(p1.y - p2.y)

            # Handle periodic boundaries
            dx = min(dx, self.width - dx)
            dy = min(dy, self.height - dy)

            return math.sqrt(dx * dx + dy * dy)

    def get_periodic_vector(self, p1: Particle, p2: Particle) -> tuple[float, float]:
        dx: float = p2.x - p1.x
        dy: float = p2.y - p1.y

        if self.boundary_type == BoundaryType.PERIODIC:
            # Handle periodic boundaries
            if abs(dx) > self.width / 2:
                dx = dx - math.copysign(self.width, dx)
            if abs(dy) > self.height / 2:
                dy = dy - math.copysign(self.height, dy)

        return dx, dy

    def create_particle_at_position(self, particle_type: str, x: float, y: float, vx: float = 0, vy: float = 0) -> Particle:
        if particle_type not in PARTICLE_TYPES:
            raise ValueError(f"Unknown particle type: {particle_type}")

        ptype: ParticleType = PARTICLE_TYPES[particle_type]

        # If speed is set and velocities are default, use random direction with fixed speed
        if self.speed is not None and vx == 0 and vy == 0:
            angle = random.uniform(0, 2 * math.pi)
            vx = self.speed * math.cos(angle)
            vy = self.speed * math.sin(angle)

        return Particle(x, y, vx, vy, ptype["radius"], ptype["color"], ptype["type"])

    def remove_particles(self, particles_to_remove: list[Particle]) -> None:
        for particle in particles_to_remove:
            if particle in self.particles:
                self.particles.remove(particle)

    def log_reaction(self, current_time: float, reactants: tuple[str, str], products: tuple[str | None, str | None]) -> None:
        # Count particles by type
        particle_counts = {}
        for particle_type in PARTICLE_TYPES.keys():
            particle_counts[particle_type] = sum(1 for p in self.particles if p.type == particle_type)

        # Log the reaction event
        log_entry = {
            'time': current_time,
            'reactant1': reactants[0],
            'reactant2': reactants[1],
            'product1': products[0],
            'product2': products[1],
            'total_particles': len(self.particles),
            **{f'count_{ptype}': count for ptype, count in particle_counts.items()}
        }
        self.reaction_log.append(log_entry)

    def save_reaction_log(self, filename: str = "reactions.csv") -> None:
        if not self.reaction_log:
            print("No reactions to save.")
            return

        with open(filename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.reaction_log[0].keys())
            writer.writeheader()
            writer.writerows(self.reaction_log)

    def handle_collision(self, p1: Particle, p2: Particle, current_time: float) -> tuple[list[Particle], list[Particle], tuple | None]:
        distance: float = self.distance_periodic(p1, p2)
        min_distance: float = p1.radius + p2.radius

        if distance < min_distance and distance > 0:
            # Track collision between particle types
            type_pair = tuple(sorted([p1.type, p2.type]))
            if type_pair not in self.collision_times:
                self.collision_times[type_pair] = []
            self.collision_times[type_pair].append(current_time)

            # Check if there's a reaction defined for this collision
            reaction_key = (p1.type, p2.type)
            reverse_reaction_key = (p2.type, p1.type)

            if reaction_key in self.reactions:
                products = self.reactions[reaction_key]
            elif reverse_reaction_key in self.reactions:
                products = self.reactions[reverse_reaction_key]
            else:
                # No reaction defined, use elastic collision
                self._handle_elastic_collision(p1, p2)
                return [], [], None

            # Apply reaction
            particles_to_remove = [p1, p2]
            particles_to_add = []

            # Store reaction info for logging after particle removal
            reaction_info = (current_time, (p1.type, p2.type), products)

            # Calculate collision center for new particle placement
            collision_x = (p1.x + p2.x) / 2
            collision_y = (p1.y + p2.y) / 2

            # Create new particles based on reaction products
            for i, product_type in enumerate(products):
                if product_type is not None:
                    # Add small random offset to avoid immediate collision of products
                    offset_x = random.uniform(-5, 5)
                    offset_y = random.uniform(-5, 5)

                    # Use speed from original particles with random direction
                    p1_speed = math.sqrt(p1.vx**2 + p1.vy**2)
                    p2_speed = math.sqrt(p2.vx**2 + p2.vy**2)
                    avg_speed = (p1_speed + p2_speed) / 2

                    angle = random.uniform(0, 2 * math.pi)
                    vx = avg_speed * math.cos(angle)
                    vy = avg_speed * math.sin(angle)

                    new_particle = self.create_particle_at_position(
                        product_type,
                        collision_x + offset_x,
                        collision_y + offset_y,
                        vx,
                        vy
                    )
                    particles_to_add.append(new_particle)

            return particles_to_remove, particles_to_add, reaction_info

        return [], [], None

    def _handle_elastic_collision(self, p1: Particle, p2: Particle) -> tuple[list[Particle], list[Particle]]:
        dx: float
        dy: float
        dx, dy = self.get_periodic_vector(p1, p2)

        # Normalize collision vector
        collision_distance: float = math.sqrt(dx * dx + dy * dy)
        if collision_distance == 0:
            return [], []

        nx: float = dx / collision_distance
        ny: float = dy / collision_distance

        # Separate particles
        min_distance: float = p1.radius + p2.radius
        overlap: float = min_distance - collision_distance
        separation: float = overlap * 0.5
        p1.x -= separation * nx
        p2.x += separation * nx
        p1.y -= separation * ny
        p2.y += separation * ny

        # Handle boundaries after separation
        if self.boundary_type == BoundaryType.PERIODIC:
            p1.x = p1.x % self.width
            p1.y = p1.y % self.height
            p2.x = p2.x % self.width
            p2.y = p2.y % self.height

        # Calculate relative velocity
        dvx: float = p2.vx - p1.vx
        dvy: float = p2.vy - p1.vy
        dvn: float = dvx * nx + dvy * ny

        # Only resolve if objects are moving towards each other
        if dvn > 0:
            return [], []

        # Calculate impulse
        impulse: float = 2 * dvn / (p1.mass + p2.mass)

        # Update velocities
        p1.vx += impulse * p2.mass * nx
        p1.vy += impulse * p2.mass * ny
        p2.vx -= impulse * p1.mass * nx
        p2.vy -= impulse * p1.mass * ny

        return [], []

    def update(self, dt: float, current_time: float) -> None:
        # Clear and populate spatial grid
        self.grid.clear()
        for particle in self.particles:
            particle.update(dt, self.width, self.height, self.boundary_type)
            self.grid.add_particle(particle)

        # Check collisions using spatial grid and collect particle changes
        checked_pairs: set[tuple[int, int]] = set()
        particles_to_remove: list[Particle] = []
        particles_to_add: list[Particle] = []
        particles_already_removed: set[int] = set()
        reactions_to_log: list = []

        for particle in self.particles:
            if id(particle) in particles_already_removed:
                continue

            nearby: list[Particle] = self.grid.get_nearby_particles(particle)
            for other in nearby:
                if particle is not other and id(other) not in particles_already_removed:
                    pair: tuple[int, int] = tuple(sorted([id(particle), id(other)]))
                    if pair not in checked_pairs:
                        checked_pairs.add(pair)
                        remove_list, add_list, reaction_info = self.handle_collision(particle, other, current_time)
                        particles_to_remove.extend(remove_list)
                        particles_to_add.extend(add_list)

                        # Store reaction for logging after particle removal
                        if reaction_info:
                            reactions_to_log.append(reaction_info)

                        # Track particles that will be removed to prevent multiple reactions
                        for p in remove_list:
                            particles_already_removed.add(id(p))

        # Apply particle changes after collision detection is complete
        self.remove_particles(particles_to_remove)
        self.particles.extend(particles_to_add)

        # Log reactions after particles have been removed/added
        for reaction_info in reactions_to_log:
            current_time, reactants, products = reaction_info
            self.log_reaction(current_time, reactants, products)

    def draw(self) -> None:
        self.screen.fill((20, 20, 20))

        for particle in self.particles:
            pygame.draw.circle(
                self.screen,
                particle.color,
                (int(particle.x), int(particle.y)),
                particle.radius,
            )

        # Draw FPS
        fps: float = self.clock.get_fps()
        font: pygame.font.Font = pygame.font.Font(None, 36)
        fps_text: pygame.Surface = font.render(
            f"FPS: {fps:.1f} | Particles: {len(self.particles)}", True, (255, 255, 255)
        )
        self.screen.blit(fps_text, (10, 10))

        pygame.display.flip()

    def run(
        self,
        max_duration: float | None = None,
        track_collision_pair: tuple[str, str] | None = None,
        relaxation_time: float = 1.0,
    ) -> int | None:
        # Relaxation phase - let particles settle without tracking collisions
        print(f"Relaxing particles for {relaxation_time} seconds...")
        relaxation_start = 0.0
        while relaxation_start < relaxation_time:
            dt: float = self.clock.tick(60) / 1000.0
            relaxation_start += dt

            # Update physics but don't track collisions
            self.grid.clear()
            for particle in self.particles:
                particle.update(dt, self.width, self.height, self.boundary_type)
                self.grid.add_particle(particle)

            # Handle collisions without timing tracking - only elastic collisions during relaxation
            checked_pairs: set[tuple[int, int]] = set()

            for particle in self.particles:
                nearby: list[Particle] = self.grid.get_nearby_particles(particle)
                for other in nearby:
                    if particle is not other:
                        pair: tuple[int, int] = tuple(sorted([id(particle), id(other)]))
                        if pair not in checked_pairs:
                            checked_pairs.add(pair)
                            # Only elastic collisions during relaxation - no particle removal
                            self._handle_collision_no_tracking(particle, other)

        # Reset collision tracking after relaxation
        self.collision_times.clear()
        print("Relaxation complete. Starting measurement...")

        # Main simulation phase
        running: bool = True
        self.start_time = 0.0
        current_time: float = 0.0

        while running:
            dt: float = self.clock.tick(60) / 1000.0
            current_time += dt

            # Check if max duration reached
            if max_duration is not None and current_time >= max_duration:
                running = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            self.update(dt, current_time)
            self.draw()

        pygame.quit()

        # Return collision count for specified pair
        if track_collision_pair is not None:
            normalized_pair = tuple(sorted(track_collision_pair))
            if normalized_pair in self.collision_times:
                return len(self.collision_times[normalized_pair])
            return 0

        return None

    def _handle_collision_no_tracking(self, p1: Particle, p2: Particle) -> None:
        """Handle collision without tracking collision times - only elastic collisions during relaxation"""
        distance: float = self.distance_periodic(p1, p2)
        min_distance: float = p1.radius + p2.radius

        if distance < min_distance and distance > 0:
            # During relaxation, only do elastic collisions (no reactions, no particle removal)
            self._handle_elastic_collision_direct(p1, p2)

    def _handle_elastic_collision_direct(self, p1: Particle, p2: Particle) -> None:
        """Direct elastic collision without returning particle lists"""
        dx: float
        dy: float
        dx, dy = self.get_periodic_vector(p1, p2)

        # Normalize collision vector
        collision_distance: float = math.sqrt(dx * dx + dy * dy)
        if collision_distance == 0:
            return

        nx: float = dx / collision_distance
        ny: float = dy / collision_distance

        # Separate particles
        min_distance: float = p1.radius + p2.radius
        overlap: float = min_distance - collision_distance
        separation: float = overlap * 0.5
        p1.x -= separation * nx
        p2.x += separation * nx
        p1.y -= separation * ny
        p2.y += separation * ny

        # Handle boundaries after separation
        if self.boundary_type == BoundaryType.PERIODIC:
            p1.x = p1.x % self.width
            p1.y = p1.y % self.height
            p2.x = p2.x % self.width
            p2.y = p2.y % self.height

        # Calculate relative velocity
        dvx: float = p2.vx - p1.vx
        dvy: float = p2.vy - p1.vy
        dvn: float = dvx * nx + dvy * ny

        # Only resolve if objects are moving towards each other
        if dvn > 0:
            return

        # Calculate impulse
        impulse: float = 2 * dvn / (p1.mass + p2.mass)

        # Update velocities
        p1.vx += impulse * p2.mass * nx
        p1.vy += impulse * p2.mass * ny
        p2.vx -= impulse * p1.mass * nx
        p2.vy -= impulse * p1.mass * ny


def hit_times(
    N: int = 1000,
    max_time: float = 2000,
    fixed_speed=50.0,
    boundary_type: BoundaryType = BoundaryType.PERIODIC,
    show_animation: bool = True,
) -> None:
    particle_type_static = "huge"
    particle_type_move = "huge"

    hit_times_list = []

    for i in range(N):
        print(f"Running simulation {i + 1}/{N}...")

        # Create collider with no particles initially
        collider = ParticleCollider(particle_counts={}, boundary_type=boundary_type)

        # Initialize particle_type_static ball at center
        #
        # NOTE:
        # finally we made it not so static, to get an exp
        # with static it looks exp-like but can be rejected as not being
        angle_static = random.uniform(0, 2 * math.pi)
        vx_static = fixed_speed * math.cos(angle_static)
        vy_static = fixed_speed * math.sin(angle_static)

        static_particle = Particle(
            x=collider.width / 2,
            y=collider.height / 2,
            vx=vx_static,
            vy=vy_static,
            radius=PARTICLE_TYPES[particle_type_static]["radius"],
            color=PARTICLE_TYPES[particle_type_static]["color"],
            particle_type=particle_type_static,
        )

        # Initialize particle_type_move ball at random position with random direction
        x = random.uniform(
            PARTICLE_TYPES[particle_type_move]["radius"],
            collider.width - PARTICLE_TYPES[particle_type_move]["radius"],
        )
        y = random.uniform(
            PARTICLE_TYPES[particle_type_move]["radius"],
            collider.height - PARTICLE_TYPES[particle_type_move]["radius"],
        )

        # Random direction with fixed speed
        angle = random.uniform(0, 2 * math.pi)
        vx = fixed_speed * math.cos(angle)
        vy = fixed_speed * math.sin(angle)

        moving_particle = Particle(
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            radius=PARTICLE_TYPES[particle_type_move]["radius"],
            color=PARTICLE_TYPES[particle_type_move]["color"],
            particle_type=particle_type_move,
        )

        collider.particles = [static_particle, moving_particle]

        # Run simulation until collision
        current_time = 0.0
        collision_detected = False

        if show_animation and i == 0:
            # Show animation for the first simulation only
            pygame.init()
            screen = pygame.display.set_mode((collider.width, collider.height))
            pygame.display.set_caption(f"Hit Time Animation - {boundary_type.value}")
            clock = pygame.time.Clock()

            running = True
            while running and current_time < max_time and not collision_detected:
                dt = clock.tick(60) / 1000.0
                current_time += dt

                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False

                # Update particles
                for particle in collider.particles:
                    particle.update(
                        dt, collider.width, collider.height, collider.boundary_type
                    )

                # Check for collision
                distance = collider.distance_periodic(static_particle, moving_particle)
                if distance < (static_particle.radius + moving_particle.radius):
                    hit_times_list.append(current_time)
                    collision_detected = True
                    print(f"Collision detected at time {current_time:.3f}s in animation!")

                # Draw
                screen.fill((20, 20, 20))
                for particle in collider.particles:
                    pygame.draw.circle(
                        screen,
                        particle.color,
                        (int(particle.x), int(particle.y)),
                        particle.radius,
                    )

                # Draw time and other info
                font = pygame.font.Font(None, 36)
                time_text = font.render(f"Time: {current_time:.2f}s", True, (255, 255, 255))
                screen.blit(time_text, (10, 10))

                boundary_text = font.render(f"Boundary: {boundary_type.value}", True, (255, 255, 255))
                screen.blit(boundary_text, (10, 50))

                pygame.display.flip()

            pygame.quit()

            if not collision_detected:
                print(f"Warning: No collision detected in animation within {max_time}s")

        else:
            # Fast simulation without animation
            dt = 1/ 60
            while current_time < max_time and not collision_detected:
                current_time += dt

                # Update particles
                for particle in collider.particles:
                    particle.update(
                        dt, collider.width, collider.height, collider.boundary_type
                    )

                # Check for collision
                distance = collider.distance_periodic(static_particle, moving_particle)
                if distance < (static_particle.radius + moving_particle.radius):
                    hit_times_list.append(current_time)
                    collision_detected = True

            if not collision_detected:
                print(
                    f"Warning: No collision detected in simulation {i + 1} within {max_time}s"
                )

    # Save to CSV
    with open("hit_times.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["hit_time"])
        for hit_time in hit_times_list:
            writer.writerow([hit_time])

    # Fit exponential distribution
    hit_times_array = np.array(hit_times_list)

    # Fit exponential distribution (scipy uses scale parameter, not rate)
    # For exponential: f(x) = (1/scale) * exp(-x/scale), where scale = 1/lambda
    loc, scale = stats.expon.fit(hit_times_array, floc=0)  # floc=0 fixes location at 0
    lambda_param = 1 / scale  # rate parameter

    # Perform Kolmogorov-Smirnov test for exponential distribution
    ks_statistic, ks_p_value = stats.kstest(
        hit_times_array, lambda x: stats.expon.cdf(x, loc=loc, scale=scale)
    )

    # Create two subplots: histogram and CDF comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Histogram with fitted exponential overlay
    n, bins, patches = ax1.hist(
        hit_times_list,
        bins=100,
        alpha=0.7,
        edgecolor="black",
        density=True,
        label="Observed data",
    )

    # Fitted exponential curve
    x_fit = np.linspace(0, max(hit_times_list), 1000)
    y_fit = stats.expon.pdf(x_fit, loc=loc, scale=scale)
    ax1.plot(
        x_fit,
        y_fit,
        "r-",
        linewidth=2,
        label=f"Fitted Exponential (λ={lambda_param:.4f})",
    )

    ax1.set_xlabel("Hit Time (seconds)")
    ax1.set_ylabel("Density")
    ax1.set_title(
        f"PDF: Hit Times with Exponential Fit\nN={len(hit_times_list)} collisions"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: CDF comparison
    # Empirical CDF
    sorted_data = np.sort(hit_times_array)
    empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax2.plot(sorted_data, empirical_cdf, "b-", linewidth=2, label="Empirical CDF")

    # Theoretical exponential CDF
    theoretical_cdf = stats.expon.cdf(sorted_data, loc=loc, scale=scale)
    ax2.plot(
        sorted_data,
        theoretical_cdf,
        "r-",
        linewidth=2,
        label=f"Exponential CDF (λ={lambda_param:.4f})",
    )

    ax2.set_xlabel("Hit Time (seconds)")
    ax2.set_ylabel("Cumulative Probability")
    ax2.set_title(f"CDF Comparison\nKS statistic = {ks_statistic:.4f}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Mean hit time: {sum(hit_times_list) / len(hit_times_list):.3f}s")
    print(f"Successful collisions: {len(hit_times_list)}/{N}")
    print(f"Fitted exponential rate parameter (λ): {lambda_param:.6f}")
    print(f"Theoretical mean (1/λ): {1 / lambda_param:.3f}s")

    print("\nKolmogorov-Smirnov test for exponential distribution:")
    print(f"KS statistic: {ks_statistic:.6f}")
    print(f"p-value: {ks_p_value:.6f}")
    if ks_p_value > 0.05:
        print("✓ Cannot reject exponential distribution hypothesis (p > 0.05)")
    else:
        print("✗ Reject exponential distribution hypothesis (p ≤ 0.05)")


def sweep(boundary_type: BoundaryType = BoundaryType.PERIODIC) -> None:
    x = []
    y = []

    # time for measuring
    T = 10
    for small_count in range(1, 1000, 100):
        particle_counts = {"small": small_count, "medium": 5}
        collider: ParticleCollider = ParticleCollider(
            particle_counts=particle_counts, boundary_type=boundary_type
        )
        collision_count = collider.run(
            max_duration=T,
            track_collision_pair=("small", "medium"),
            relaxation_time=2.0,
        )
        x.append(small_count)
        if collision_count is not None:
            y.append(collision_count / particle_counts["medium"])
        else:
            y.append(0)
    # Save to CSV
    with open("collision_data.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["num_medium_particles", "collision_count"])
        for i, time_val in zip(x, y):
            writer.writerow([i, time_val])

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, "bo-", linewidth=2, markersize=8)
    plt.xlabel("Number of Medium Particles")
    plt.ylabel("Total Collision Count")
    plt.title("Collision Count vs Number of Medium Particles")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_reactions(species: list[str] | None = None, filename: str = "reactions.csv") -> None:
    try:
        # Read the CSV file
        data = []
        with open(filename, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)

        if not data:
            print("No reaction data found in CSV file.")
            return

        # Convert to appropriate types
        times = [float(row['time']) for row in data]
        total_particles = [int(row['total_particles']) for row in data]

        # Get all particle types from the data
        all_particle_types = []
        for key in data[0].keys():
            if key.startswith('count_'):
                all_particle_types.append(key.replace('count_', ''))

        # Use specified species or all available types
        if species is None:
            plot_species = all_particle_types
        else:
            # Filter to only species that exist in the data
            plot_species = [s for s in species if f'count_{s}' in data[0].keys()]
            if not plot_species:
                print(f"None of the specified species {species} found in data.")
                print(f"Available species: {all_particle_types}")
                return

        # Create single figure for all species
        plt.figure(figsize=(12, 8))

        # Plot individual particle types using their actual colors
        for ptype in plot_species:
            counts = [int(row[f'count_{ptype}']) for row in data]
            # Get the particle color and convert to matplotlib format (0-1 range)
            if ptype in PARTICLE_TYPES:
                particle_color = PARTICLE_TYPES[ptype]['color']
                plot_color = tuple(c / 255.0 for c in particle_color)  # Convert RGB 0-255 to 0-1
            else:
                plot_color = 'gray'  # Fallback color

            plt.plot(times, counts, color=plot_color, linewidth=2,
                    label=f'{ptype.upper()} particles', marker='o', markersize=3, alpha=0.8)

        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Particle Count', fontsize=12)
        plt.title('Particle Counts Over Time', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        plt.tight_layout()
        plt.show()

        # Print summary statistics
        print(f"Total reactions logged: {len(data)}")
        print(f"Time range: {min(times):.2f} - {max(times):.2f} seconds")
        print(f"Final particle counts:")
        final_data = data[-1]
        for ptype in plot_species:
            print(f"  {ptype.upper()}: {final_data[f'count_{ptype}']}")
        print(f"  Total: {final_data['total_particles']}")

    except FileNotFoundError:
        print(f"CSV file '{filename}' not found. Run a simulation with reactions first.")
    except Exception as e:
        print(f"Error reading CSV file: {e}")


def reactions(boundary_type: BoundaryType = BoundaryType.PERIODIC) -> None:
    width = 400
    height = 400
    particle_counts = {"A": 51, "B": 49,}

    # width = 2000
    # height = 2000
    # particle_counts = {"A": 551, "B": 549,}

    reactions: Reaction = {
        ("A", "B"): (None, None),
        ("B", "A"): (None, None),  # Symmetric reaction
    }

    # Create collider with reactions
    collider: ParticleCollider = ParticleCollider(
        particle_counts=particle_counts,
        boundary_type=boundary_type,
        reactions=reactions,
        width=width,
        height=height,
        speed = 100,
    )

    # Log initial state before simulation
    collider.log_reaction(0.0, ("init", "init"), (None, None))

    # Run simulation
    collider.run(max_duration=20)

    # Save reaction log to CSV
    collider.save_reaction_log("reactions.csv")
    print("Reaction data saved to reactions.csv")


if __name__ == "__main__":
    # boundary_type = BoundaryType.PERIODIC
    boundary_type = BoundaryType.REFLECTIVE

    # Example 0: video
    # particle_counts = {"small": 3, "huge": 1, "medium": 10}
    # collider: ParticleCollider = ParticleCollider(particle_counts=particle_counts, boundary_type=boundary_type)
    # collider.run(max_duration=20.0, track_collision_pair=("small", "huge"))

    # Example 1: show almost exp hit time
    # hit_times(boundary_type=boundary_type)

    # Example 2: show linearity of number of collisions
    # sweep(boundary_type=boundary_type)

    # Example 3: reaction system
    reactions(boundary_type=boundary_type)
    plot_reactions(["A", "B"])
