"""
UZB 438E - Robotic Control Systems Final Project
Real Traffic Car Following Scenario in CARLA 0.9.16

Run:
    1) Start CARLA 0.9.16 server.
    2) python car_following_real_traffic_final.py
"""

import carla
import math
import random
import time
import traceback


# ============================================================
# Scenario parameters
# ============================================================

MAP_NAME = "Town03"
RANDOM_SEED = 42

FIXED_DELTA_SECONDS = 0.05
SIM_DURATION_SECONDS = 175.0

DESIRED_DISTANCE_M = 14.0
INITIAL_EGO_GAP_M = 25.0

ROUTE_STEP_M = 3.0
ROUTE_LENGTH_POINTS = 420

LEAD_CRUISE_SPEED_KMH = 28.0
LEAD_CURVE_SPEED_KMH = 16.0
EGO_MAX_SPEED_KMH = 48.0

LOOKAHEAD_LEAD_M = 10.0
LOOKAHEAD_EGO_M = 9.0

ROUTE_TRAFFIC_CARS = 6
CROSS_TRAFFIC_CARS = 14
PEDESTRIANS = 12

TRAFFIC_LIGHT_MAX_WAIT_SECONDS = 5.0
EGO_IGNORE_LIGHT_AFTER_TIMEOUT_SECONDS = 5.0
LEAD_IGNORE_LIGHT_AFTER_TIMEOUT_SECONDS = 5.0

LEAD_WAIT_FOR_EGO_GAP_M = 42.0
LEAD_WAIT_SPEED_KMH = 6.0

PRINT_EVERY_N_TICKS = 10


# ============================================================
# Basic helpers
# ============================================================

def clamp(value, low, high):
    return max(low, min(value, high))


def safe_tick(world):
    try:
        return world.tick()
    except RuntimeError:
        time.sleep(0.05)
        return world.tick()


def distance_2d(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def distance_3d(a, b):
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2 +
        (a.z - b.z) ** 2
    )


def get_speed_mps(vehicle):
    v = vehicle.get_velocity()
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def get_speed_kmh(vehicle):
    return get_speed_mps(vehicle) * 3.6


def normalize_angle_rad(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def get_local_position(reference_actor, target_location):
    ref_tf = reference_actor.get_transform()
    ref_location = ref_tf.location
    forward = ref_tf.get_forward_vector()

    dx = target_location.x - ref_location.x
    dy = target_location.y - ref_location.y

    longitudinal = dx * forward.x + dy * forward.y
    lateral = dx * (-forward.y) + dy * forward.x

    return longitudinal, lateral


# ============================================================
# PID controllers
# ============================================================

class PIDController:
    def __init__(self, kp, ki, kd, dt, integral_limit=10.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral_limit = integral_limit

        self.integral = 0.0
        self.previous_error = 0.0
        self.first_step = True

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.first_step = True

    def run_step(self, error):
        self.integral += error * self.dt
        self.integral = clamp(
            self.integral,
            -self.integral_limit,
            self.integral_limit,
        )

        if self.first_step:
            derivative = 0.0
            self.first_step = False
        else:
            derivative = (error - self.previous_error) / self.dt

        self.previous_error = error

        return (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )


class SpeedController:
    def __init__(self, dt):
        self.pid = PIDController(
            kp=0.10,
            ki=0.012,
            kd=0.006,
            dt=dt,
            integral_limit=18.0,
        )

    def reset(self):
        self.pid.reset()

    def run_step(
        self,
        vehicle,
        target_speed_kmh,
        max_throttle=0.45,
        max_brake=0.80,
    ):
        current_speed = get_speed_kmh(vehicle)
        error = target_speed_kmh - current_speed
        command = self.pid.run_step(error)

        if command >= 0.0:
            throttle = clamp(command, 0.0, max_throttle)
            brake = 0.0
        else:
            throttle = 0.0
            brake = clamp(-command, 0.0, max_brake)

        if brake > 0.03:
            throttle = 0.0

        return throttle, brake


# ============================================================
# Route generation
# ============================================================

def waypoint_key(wp):
    return (wp.road_id, wp.section_id, wp.lane_id, int(wp.s // 5.0))


def choose_city_next(current_wp, previous_wp, visited):
    next_wps = current_wp.next(ROUTE_STEP_M)

    if not next_wps:
        return None

    current_yaw = math.radians(current_wp.transform.rotation.yaw)

    scored = []

    for wp in next_wps:
        next_yaw = math.radians(wp.transform.rotation.yaw)
        yaw_diff = abs(normalize_angle_rad(next_yaw - current_yaw))
        yaw_diff_deg = math.degrees(yaw_diff)

        key = waypoint_key(wp)

        score = 0.0

        if key not in visited:
            score += 8.0
        else:
            score -= 12.0

        if wp.is_junction:
            score += 10.0

        if 8.0 <= yaw_diff_deg <= 55.0:
            score += 5.0

        if yaw_diff_deg > 85.0:
            score -= 10.0

        score += random.uniform(-2.0, 2.0)

        scored.append((score, wp))

    scored.sort(key=lambda item: item[0], reverse=True)

    return scored[0][1]


def generate_city_route(start_wp, max_points=420):
    route = [start_wp]
    current = start_wp
    previous = None
    visited = set()
    visited.add(waypoint_key(start_wp))

    for _ in range(max_points - 1):
        nxt = choose_city_next(current, previous, visited)

        if nxt is None:
            break

        route.append(nxt)
        visited.add(waypoint_key(nxt))

        previous = current
        current = nxt

    return route


def route_total_length(route):
    if len(route) < 2:
        return 0.0

    total = 0.0

    for i in range(1, len(route)):
        total += distance_2d(
            route[i - 1].transform.location,
            route[i].transform.location,
        )

    return total


def count_junction_points(route):
    return sum(1 for wp in route if wp.is_junction)


def find_closest_route_index(
    vehicle,
    route,
    last_index,
    search_back=10,
    search_forward=55,
):
    location = vehicle.get_location()

    start = max(0, last_index - search_back)
    end = min(len(route), last_index + search_forward)

    best_index = last_index
    best_distance = float("inf")

    for i in range(start, end):
        d = distance_2d(location, route[i].transform.location)

        if d < best_distance:
            best_distance = d
            best_index = i

    return best_index


def target_index_from_lookahead(route, closest_index, lookahead_m):
    steps = max(2, int(lookahead_m / ROUTE_STEP_M))
    return min(len(route) - 1, closest_index + steps)


def pure_pursuit_steer(vehicle, target_location, gain=1.20, max_steer=0.62):
    transform = vehicle.get_transform()
    location = transform.location
    heading = math.radians(transform.rotation.yaw)

    target_angle = math.atan2(
        target_location.y - location.y,
        target_location.x - location.x,
    )

    heading_error = normalize_angle_rad(target_angle - heading)
    steer = gain * heading_error

    return clamp(steer, -max_steer, max_steer)


def curve_angle_ahead(route, index, near_offset=2, far_offset=13):
    if not route:
        return 0.0

    i1 = min(len(route) - 1, index + near_offset)
    i2 = min(len(route) - 1, index + far_offset)

    yaw1 = math.radians(route[i1].transform.rotation.yaw)
    yaw2 = math.radians(route[i2].transform.rotation.yaw)

    return abs(math.degrees(normalize_angle_rad(yaw2 - yaw1)))


# ============================================================
# Traffic light helpers
# ============================================================

def configure_traffic_lights(world):
    for light in world.get_actors().filter("traffic.traffic_light"):
        try:
            light.freeze(False)
            light.set_green_time(10.0)
            light.set_yellow_time(2.5)
            light.set_red_time(6.0)
        except Exception:
            pass


def handle_traffic_light_with_timeout(
    vehicle,
    vehicle_name,
    light_wait_memory,
    simulation_time,
):
    """
    Red/yellow light => stop.
    If the vehicle waits too long, force the traffic light to green.
    """

    try:
        if not vehicle.is_at_traffic_light():
            light_wait_memory[vehicle_name] = None
            return None, False

        traffic_light = vehicle.get_traffic_light()

        if traffic_light is None:
            light_wait_memory[vehicle_name] = None
            return None, False

        state = traffic_light.get_state()

        if state == carla.TrafficLightState.Green:
            light_wait_memory[vehicle_name] = None
            return None, False

        if state == carla.TrafficLightState.Red:
            state_name = "RED"
        elif state == carla.TrafficLightState.Yellow:
            state_name = "YELLOW"
        else:
            light_wait_memory[vehicle_name] = None
            return None, False

        if light_wait_memory.get(vehicle_name) is None:
            light_wait_memory[vehicle_name] = simulation_time

        waited_time = simulation_time - light_wait_memory[vehicle_name]

        if waited_time >= TRAFFIC_LIGHT_MAX_WAIT_SECONDS:
            try:
                traffic_light.set_state(carla.TrafficLightState.Green)
                traffic_light.freeze(True)
                light_wait_memory[vehicle_name] = None

                print(
                    f"[INFO] {vehicle_name} waited {waited_time:.1f}s. "
                    f"Traffic light forced GREEN."
                )

                return None, True

            except Exception:
                light_wait_memory[vehicle_name] = None
                return None, True

        return state_name, False

    except Exception:
        light_wait_memory[vehicle_name] = None
        return None, False


# ============================================================
# Risk detection  (FIX: junction-aware front actor detection)
# ============================================================

def get_front_actor(
    vehicle,
    world,
    world_map,
    ignore_ids=None,
    max_distance=40.0,      # increased from 30.0
    lateral_limit=3.2,      # increased from 2.8
):
    if ignore_ids is None:
        ignore_ids = set()

    vehicle_location = vehicle.get_location()

    vehicle_wp = world_map.get_waypoint(
        vehicle_location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    if vehicle_wp is None:
        return None, max_distance

    vehicle_in_junction = vehicle_wp.is_junction

    closest_actor = None
    closest_distance = max_distance

    for actor in world.get_actors().filter("vehicle.*"):
        if actor.id == vehicle.id:
            continue

        if actor.id in ignore_ids:
            continue

        try:
            actor_location = actor.get_location()

            longitudinal, lateral = get_local_position(vehicle, actor_location)

            # Must be in front and within lateral bounds
            if not (longitudinal > 0.5 and abs(lateral) < lateral_limit):
                continue

            if longitudinal >= closest_distance:
                continue

            # FIX: If either vehicle is in a junction, skip road/lane ID check
            # and rely purely on geometric position — road_id changes constantly
            # at junctions and causes NPC vehicles to be missed.
            if vehicle_in_junction:
                closest_distance = longitudinal
                closest_actor = actor
                continue

            actor_wp = world_map.get_waypoint(
                actor_location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )

            if actor_wp is None:
                continue

            if actor_wp.is_junction:
                # Actor is entering/leaving junction — include it
                closest_distance = longitudinal
                closest_actor = actor
                continue

            same_road = actor_wp.road_id == vehicle_wp.road_id
            same_lane = actor_wp.lane_id == vehicle_wp.lane_id

            if not same_road or not same_lane:
                continue

            closest_distance = longitudinal
            closest_actor = actor

        except Exception:
            pass

    return closest_actor, closest_distance


def pedestrian_ahead(vehicle, walkers, radius=9.0):
    for walker in walkers:
        try:
            walker_location = walker.get_location()
            longitudinal, lateral = get_local_position(vehicle, walker_location)

            if 0.5 < longitudinal < radius and abs(lateral) < 2.2:
                distance = distance_3d(vehicle.get_location(), walker_location)
                return True, distance

        except Exception:
            pass

    return False, 999.0


def emergency_gap_brake(gap):
    brake = 0.50 + (DESIRED_DISTANCE_M - gap) / DESIRED_DISTANCE_M
    return clamp(brake, 0.50, 1.0)


# ============================================================
# Spawn helpers
# ============================================================

def prepare_vehicle_blueprint(bp_lib, pattern, color, role_name):
    bp = bp_lib.find(pattern)

    if bp.has_attribute("color"):
        bp.set_attribute("color", color)

    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", role_name)

    return bp


def draw_route(world, route, every=6, life_time=180.0):
    for i in range(0, len(route), every):
        loc = route[i].transform.location + carla.Location(z=0.25)

        world.debug.draw_point(
            loc,
            size=0.08,
            color=carla.Color(0, 255, 0),
            life_time=life_time,
        )


def is_far_from_actors(location, actors, min_distance):
    for actor in actors:
        try:
            if distance_2d(location, actor.get_location()) < min_distance:
                return False
        except Exception:
            pass

    return True


def is_far_from_locations(location, locations, min_distance):
    for loc in locations:
        if distance_2d(location, loc) < min_distance:
            return False

    return True


def find_traffic_start(world_map, spawn_points):
    candidates = list(spawn_points)
    random.shuffle(candidates)

    best_candidate = None
    best_score = -999999.0

    for sp in candidates:
        start_wp = world_map.get_waypoint(
            sp.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if start_wp is None:
            continue

        previous = start_wp.previous(INITIAL_EGO_GAP_M)

        if not previous:
            continue

        route = generate_city_route(start_wp, ROUTE_LENGTH_POINTS)

        if len(route) < 220:
            continue

        length = route_total_length(route)
        junction_count = count_junction_points(route)

        if length < 320.0:
            continue

        score = length * 0.02 + junction_count * 3.0

        if junction_count < 12:
            score -= 80.0

        if score > best_score:
            best_score = score
            best_candidate = (start_wp, previous[0], route)

    if best_candidate is None:
        raise RuntimeError("Could not find a route with enough traffic/junction interaction.")

    return best_candidate


def get_vehicle_blueprints(bp_lib, wheels=4):
    return [
        bp for bp in bp_lib.filter("vehicle.*")
        if bp.has_attribute("number_of_wheels")
        and bp.get_attribute("number_of_wheels").as_int() == wheels
    ]


def setup_npc_vehicle(actor, traffic_manager, speed_diff=20.0):
    actor.set_autopilot(True, traffic_manager.get_port())

    traffic_manager.vehicle_percentage_speed_difference(actor, speed_diff)
    traffic_manager.distance_to_leading_vehicle(actor, random.uniform(8.0, 14.0))
    traffic_manager.auto_lane_change(actor, True)
    traffic_manager.ignore_lights_percentage(actor, 0.0)
    traffic_manager.ignore_signs_percentage(actor, 0.0)
    traffic_manager.ignore_walkers_percentage(actor, 0.0)
    traffic_manager.ignore_vehicles_percentage(actor, 0.0)

    try:
        traffic_manager.update_vehicle_lights(actor, True)
    except Exception:
        pass


def spawn_route_traffic(
    world,
    bp_lib,
    route,
    traffic_manager,
    protected_actors,
    count=6,
):
    car_bps = get_vehicle_blueprints(bp_lib, wheels=4)

    if not car_bps:
        return []

    spawned = []

    candidate_indices = [
        45,
        70,
        95,
        125,
        155,
        190,
        230,
        270,
    ]

    random.shuffle(candidate_indices)

    for idx in candidate_indices:
        if len(spawned) >= count:
            break

        if idx >= len(route):
            continue

        wp = route[idx]

        if wp.is_junction:
            continue

        tf = wp.transform
        tf.location.z += 0.45

        if not is_far_from_actors(tf.location, protected_actors + spawned, 18.0):
            continue

        bp = random.choice(car_bps)

        if bp.has_attribute("color"):
            bp.set_attribute(
                "color",
                random.choice(bp.get_attribute("color").recommended_values),
            )

        actor = world.try_spawn_actor(bp, tf)

        if actor is None:
            continue

        setup_npc_vehicle(
            actor,
            traffic_manager,
            speed_diff=random.uniform(25.0, 45.0),
        )

        spawned.append(actor)

    return spawned


def spawn_cross_traffic(
    world,
    bp_lib,
    world_map,
    spawn_points,
    route,
    traffic_manager,
    protected_actors,
    count=14,
):
    car_bps = get_vehicle_blueprints(bp_lib, wheels=4)
    moto_bps = get_vehicle_blueprints(bp_lib, wheels=2)
    all_bps = car_bps + moto_bps

    if not all_bps:
        return []

    junction_locations = [
        wp.transform.location
        for wp in route
        if wp.is_junction
    ]

    if not junction_locations:
        junction_locations = [
            wp.transform.location
            for wp in route[40:220:20]
        ]

    spawned = []

    candidates = list(spawn_points)
    random.shuffle(candidates)

    route_locations = [
        wp.transform.location
        for wp in route[0:260:8]
    ]

    for sp in candidates:
        if len(spawned) >= count:
            break

        near_junction = any(
            distance_2d(sp.location, loc) < 85.0
            for loc in junction_locations
        )

        if not near_junction:
            continue

        if not is_far_from_locations(sp.location, route_locations[:8], 35.0):
            continue

        if not is_far_from_actors(sp.location, protected_actors + spawned, 16.0):
            continue

        bp = random.choice(all_bps)

        if bp.has_attribute("color"):
            bp.set_attribute(
                "color",
                random.choice(bp.get_attribute("color").recommended_values),
            )

        actor = world.try_spawn_actor(bp, sp)

        if actor is None:
            continue

        setup_npc_vehicle(
            actor,
            traffic_manager,
            speed_diff=random.uniform(5.0, 25.0),
        )

        spawned.append(actor)

    return spawned


def spawn_pedestrians_near_route(
    world,
    bp_lib,
    route,
    n=12,
):
    walker_bps = bp_lib.filter("walker.pedestrian.*")
    controller_bp = bp_lib.find("controller.ai.walker")

    walkers = []
    controllers = []

    route_locations = [
        wp.transform.location
        for wp in route[20:260:10]
    ]

    spawn_locations = []

    for _ in range(n * 35):
        loc = world.get_random_location_from_navigation()

        if loc is None:
            continue

        near_route = any(
            10.0 <= distance_2d(loc, rloc) <= 75.0
            for rloc in route_locations
        )

        if near_route:
            spawn_locations.append(carla.Transform(loc))

        if len(spawn_locations) >= n:
            break

    for spawn_tf in spawn_locations[:n]:
        bp = random.choice(walker_bps)

        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")

        walker = world.try_spawn_actor(bp, spawn_tf)

        if walker:
            walkers.append(walker)

    safe_tick(world)

    for walker in walkers:
        controller = world.try_spawn_actor(
            controller_bp,
            carla.Transform(),
            attach_to=walker,
        )

        if controller:
            controllers.append(controller)

    safe_tick(world)

    for controller in controllers:
        try:
            controller.start()

            destination = world.get_random_location_from_navigation()

            if destination:
                controller.go_to_location(destination)

            controller.set_max_speed(random.uniform(0.6, 1.3))

        except Exception:
            pass

    safe_tick(world)

    return walkers, controllers


def attach_collision_sensor(world, bp_lib, vehicle, name, collision_log):
    sensor_bp = bp_lib.find("sensor.other.collision")

    sensor = world.spawn_actor(
        sensor_bp,
        carla.Transform(),
        attach_to=vehicle,
    )

    def on_collision(event):
        other = event.other_actor
        impulse = event.normal_impulse

        intensity = math.sqrt(
            impulse.x * impulse.x +
            impulse.y * impulse.y +
            impulse.z * impulse.z
        )

        msg = f"{name} collided with {other.type_id}, intensity={intensity:.1f}"
        collision_log.append(msg)
        print("[COLLISION]", msg)

    sensor.listen(on_collision)

    return sensor


# ============================================================
# Lead control
# ============================================================

def lead_control(
    lead,
    ego,
    route,
    route_state,
    speed_controller,
    world,
    world_map,
    walkers,
    simulation_time,
    light_wait_memory,
):
    route_state["lead_index"] = find_closest_route_index(
        lead,
        route,
        route_state["lead_index"],
    )

    lead_index = route_state["lead_index"]

    target_idx = target_index_from_lookahead(
        route,
        lead_index,
        LOOKAHEAD_LEAD_M,
    )

    target_location = route[target_idx].transform.location

    steer = pure_pursuit_steer(
        lead,
        target_location,
        gain=1.10,
        max_steer=0.58,
    )

    # Traffic light logic with fallback ignore mode
    if simulation_time < light_wait_memory.get("lead_ignore_light_until", 0.0):
        light_state = None
    else:
        light_state, forced_green = handle_traffic_light_with_timeout(
            lead,
            "lead",
            light_wait_memory,
            simulation_time,
        )

    if light_state:
        if light_wait_memory.get("lead_red_start") is None:
            light_wait_memory["lead_red_start"] = simulation_time

        lead_red_wait = simulation_time - light_wait_memory["lead_red_start"]

        if lead_red_wait >= LEAD_IGNORE_LIGHT_AFTER_TIMEOUT_SECONDS:
            light_wait_memory["lead_ignore_light_until"] = simulation_time + 6.0
            light_wait_memory["lead_red_start"] = None

            print(
                f"[INFO] Lead waited {lead_red_wait:.1f}s at {light_state}. "
                f"Lead will ignore this light for 6 seconds."
            )

        else:
            speed_controller.reset()

            control = carla.VehicleControl(
                throttle=0.0,
                brake=0.90,
                steer=float(steer),
            )

            return control, 0.0, 0.0, f"LIGHT_{light_state}"
    else:
        light_wait_memory["lead_red_start"] = None

    pedestrian_detected, pedestrian_distance = pedestrian_ahead(
        lead,
        walkers,
        radius=9.0,
    )

    if pedestrian_detected:
        speed_controller.reset()

        brake = clamp(
            0.45 + (9.0 - pedestrian_distance) / 9.0,
            0.45,
            1.0,
        )

        control = carla.VehicleControl(
            throttle=0.0,
            brake=float(brake),
            steer=float(steer),
        )

        return control, 0.0, 0.0, f"PEDESTRIAN_{pedestrian_distance:.1f}m"

    # FIX: increased max_distance 30 -> 40, lateral_limit 2.8 -> 3.2
    front_actor, front_distance = get_front_actor(
        lead,
        world,
        world_map,
        ignore_ids=set(),
        max_distance=40.0,
        lateral_limit=3.2,
    )

    if front_actor is not None:
        front_speed = get_speed_kmh(front_actor)

        # FIX: emergency stop threshold increased 7.0 -> 10.0
        if front_distance < 10.0:
            speed_controller.reset()

            control = carla.VehicleControl(
                throttle=0.0,
                brake=1.0,
                steer=float(steer),
            )

            return control, 0.0, 0.0, f"FRONT_STOP_{front_distance:.1f}m"

        # FIX: slow-follow threshold increased 20.0 -> 28.0
        if front_distance < 28.0:
            target_speed = min(
                LEAD_CRUISE_SPEED_KMH,
                max(5.0, front_speed - 1.0),
            )

            throttle, brake = speed_controller.run_step(
                lead,
                target_speed,
                max_throttle=0.30,
                max_brake=0.80,
            )

            brake = max(
                brake,
                clamp((28.0 - front_distance) / 28.0 * 0.45, 0.0, 0.45),
            )

            control = carla.VehicleControl(
                throttle=float(throttle),
                brake=float(brake),
                steer=float(steer),
            )

            return control, target_speed, 0.0, f"TRAFFIC_{front_distance:.1f}m"

    gap_to_ego = distance_2d(
        lead.get_location(),
        ego.get_location(),
    )

    if gap_to_ego > LEAD_WAIT_FOR_EGO_GAP_M:
        target_speed = LEAD_WAIT_SPEED_KMH

        throttle, brake = speed_controller.run_step(
            lead,
            target_speed,
            max_throttle=0.20,
            max_brake=0.65,
        )

        if get_speed_kmh(lead) > target_speed + 3.0:
            throttle = 0.0
            brake = max(brake, 0.45)

        control = carla.VehicleControl(
            throttle=float(throttle),
            brake=float(brake),
            steer=float(steer),
        )

        return control, target_speed, 0.0, f"WAIT_EGO_{gap_to_ego:.1f}m"

    curve = curve_angle_ahead(route, lead_index)

    if curve > 22.0:
        target_speed = 14.0
    elif curve > 14.0:
        target_speed = LEAD_CURVE_SPEED_KMH
    elif curve > 8.0:
        target_speed = 22.0
    else:
        target_speed = LEAD_CRUISE_SPEED_KMH

    remaining = len(route) - 1 - lead_index

    if remaining < 30:
        target_speed = min(target_speed, 14.0)

    if remaining < 12:
        target_speed = 0.0

    throttle, brake = speed_controller.run_step(
        lead,
        target_speed,
        max_throttle=0.42,
        max_brake=0.78,
    )

    if target_speed <= 0.1:
        throttle = 0.0
        brake = 1.0

    control = carla.VehicleControl(
        throttle=float(throttle),
        brake=float(brake),
        steer=float(steer),
    )

    return control, target_speed, curve, "CRUISE"


# ============================================================
# Ego control
# ============================================================

def ego_control(
    ego,
    lead,
    route,
    route_state,
    distance_pid,
    speed_controller,
    world,
    world_map,
    walkers,
    simulation_time,
    light_wait_memory,
):
    route_state["ego_index"] = find_closest_route_index(
        ego,
        route,
        route_state["ego_index"],
    )

    ego_index = route_state["ego_index"]

    target_idx = target_index_from_lookahead(
        route,
        ego_index,
        LOOKAHEAD_EGO_M,
    )

    target_location = route[target_idx].transform.location

    steer = pure_pursuit_steer(
        ego,
        target_location,
        gain=1.16,
        max_steer=0.60,
    )

    gap = distance_2d(
        ego.get_location(),
        lead.get_location(),
    )

    # Traffic light logic with fallback ignore mode
    if simulation_time < light_wait_memory.get("ego_ignore_light_until", 0.0):
        light_state = None
    else:
        light_state, forced_green = handle_traffic_light_with_timeout(
            ego,
            "ego",
            light_wait_memory,
            simulation_time,
        )

    if light_state:
        if light_wait_memory.get("ego_red_start") is None:
            light_wait_memory["ego_red_start"] = simulation_time

        ego_red_wait = simulation_time - light_wait_memory["ego_red_start"]

        if ego_red_wait >= EGO_IGNORE_LIGHT_AFTER_TIMEOUT_SECONDS:
            light_wait_memory["ego_ignore_light_until"] = simulation_time + 6.0
            light_wait_memory["ego_red_start"] = None

            print(
                f"[INFO] Ego waited {ego_red_wait:.1f}s at {light_state}. "
                f"Ego will ignore this light for 6 seconds."
            )

        else:
            distance_pid.reset()
            speed_controller.reset()

            control = carla.VehicleControl(
                throttle=0.0,
                brake=0.90,
                steer=float(steer),
            )

            return control, 0.0, gap, f"LIGHT_{light_state}"
    else:
        light_wait_memory["ego_red_start"] = None

    pedestrian_detected, pedestrian_distance = pedestrian_ahead(
        ego,
        walkers,
        radius=9.0,
    )

    if pedestrian_detected:
        distance_pid.reset()
        speed_controller.reset()

        brake = clamp(
            0.45 + (9.0 - pedestrian_distance) / 9.0,
            0.45,
            1.0,
        )

        control = carla.VehicleControl(
            throttle=0.0,
            brake=float(brake),
            steer=float(steer),
        )

        return control, 0.0, gap, f"PEDESTRIAN_{pedestrian_distance:.1f}m"

    # FIX: increased max_distance 28 -> 40, lateral_limit default -> 3.2
    front_actor, front_distance = get_front_actor(
        ego,
        world,
        world_map,
        ignore_ids=set(),
        max_distance=40.0,
        lateral_limit=3.2,
    )

    if front_actor is not None:
        front_speed = get_speed_kmh(front_actor)

        # FIX: emergency stop threshold increased 6.0 -> 10.0
        if front_distance < 10.0:
            distance_pid.reset()
            speed_controller.reset()

            control = carla.VehicleControl(
                throttle=0.0,
                brake=1.0,
                steer=float(steer),
            )

            return control, 0.0, gap, f"FRONT_STOP_{front_distance:.1f}m"

        # FIX: slow-follow threshold increased 15.0 -> 22.0
        if front_actor.id != lead.id and front_distance < 22.0:
            distance_pid.reset()

            target_speed = min(
                get_speed_kmh(lead),
                max(4.0, front_speed - 1.5),
            )

            throttle, brake = speed_controller.run_step(
                ego,
                target_speed,
                max_throttle=0.28,
                max_brake=0.82,
            )

            brake = max(
                brake,
                clamp((22.0 - front_distance) / 22.0 * 0.50, 0.0, 0.50),
            )

            control = carla.VehicleControl(
                throttle=float(throttle),
                brake=float(brake),
                steer=float(steer),
            )

            return control, target_speed, gap, f"TRAFFIC_{front_distance:.1f}m"

    lead_speed = get_speed_kmh(lead)
    ego_speed = get_speed_kmh(ego)

    gap_error = gap - DESIRED_DISTANCE_M
    speed_correction = distance_pid.run_step(gap_error)

    target_speed = lead_speed + speed_correction

    if gap < DESIRED_DISTANCE_M - 5.0:
        distance_pid.reset()
        speed_controller.reset()

        control = carla.VehicleControl(
            throttle=0.0,
            brake=float(emergency_gap_brake(gap)),
            steer=float(steer),
        )

        return control, 0.0, gap, "EMERGENCY_GAP"

    if gap < DESIRED_DISTANCE_M - 2.0:
        target_speed = min(target_speed, lead_speed * 0.65)

    if gap > DESIRED_DISTANCE_M + 12.0:
        target_speed = max(target_speed, lead_speed + 8.0)

    curve = curve_angle_ahead(route, ego_index)

    if curve > 22.0:
        target_speed = min(target_speed, 18.0)
    elif curve > 14.0:
        target_speed = min(target_speed, 24.0)
    elif curve > 8.0:
        target_speed = min(target_speed, 30.0)

    target_speed = clamp(target_speed, 0.0, EGO_MAX_SPEED_KMH)

    throttle, brake = speed_controller.run_step(
        ego,
        target_speed,
        max_throttle=0.55,
        max_brake=0.82,
    )

    if abs(gap_error) < 1.2 and ego_speed > lead_speed + 1.5:
        throttle = 0.0
        brake = max(brake, 0.08)

    mode = "FOLLOW_PID"

    if gap > DESIRED_DISTANCE_M + 10.0:
        mode = "CATCH_UP"
    elif gap < DESIRED_DISTANCE_M - 1.5:
        mode = "SLOW_DOWN"

    control = carla.VehicleControl(
        throttle=float(throttle),
        brake=float(brake),
        steer=float(steer),
    )

    return control, target_speed, gap, mode


# ============================================================
# Main
# ============================================================

def main():
    random.seed(RANDOM_SEED)

    world = None
    original_settings = None
    traffic_manager = None

    actors = []
    walkers = []
    controllers = []
    collision_log = []

    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(60.0)

        print("[INFO] Connected to CARLA:", client.get_server_version())
        print(f"[INFO] Loading world: {MAP_NAME}")

        world = client.load_world(MAP_NAME)
        time.sleep(2.0)

        world_map = world.get_map()
        bp_lib = world.get_blueprint_library()
        spawn_points = world_map.get_spawn_points()

        original_settings = world.get_settings()

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        world.apply_settings(settings)

        safe_tick(world)

        world.set_weather(carla.WeatherParameters.ClearNoon)
        configure_traffic_lights(world)

        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_global_distance_to_leading_vehicle(9.0)
        traffic_manager.global_percentage_speed_difference(10.0)

        safe_tick(world)

        start_wp, ego_start_wp, route = find_traffic_start(
            world_map,
            spawn_points,
        )

        draw_route(world, route)

        lead_bp = prepare_vehicle_blueprint(
            bp_lib,
            "vehicle.tesla.model3",
            "255,0,0",
            "lead_vehicle",
        )

        ego_bp = prepare_vehicle_blueprint(
            bp_lib,
            "vehicle.audi.a2",
            "0,0,255",
            "ego_vehicle",
        )

        lead_tf = start_wp.transform
        lead_tf.location.z += 0.45

        ego_tf = ego_start_wp.transform
        ego_tf.location.z += 0.45

        lead = world.try_spawn_actor(lead_bp, lead_tf)

        if lead is None:
            raise RuntimeError("Lead vehicle could not be spawned.")

        actors.append(lead)
        safe_tick(world)

        ego = world.try_spawn_actor(ego_bp, ego_tf)

        if ego is None:
            raise RuntimeError("Ego vehicle could not be spawned.")

        actors.append(ego)
        safe_tick(world)

        lead_collision_sensor = attach_collision_sensor(
            world,
            bp_lib,
            lead,
            "Lead",
            collision_log,
        )

        ego_collision_sensor = attach_collision_sensor(
            world,
            bp_lib,
            ego,
            "Ego",
            collision_log,
        )

        actors.extend([lead_collision_sensor, ego_collision_sensor])

        print("[INFO] Spawning traffic vehicles on the main route...")

        route_traffic = spawn_route_traffic(
            world,
            bp_lib,
            route,
            traffic_manager,
            protected_actors=[lead, ego],
            count=ROUTE_TRAFFIC_CARS,
        )

        actors.extend(route_traffic)
        print(f"[INFO] Route traffic vehicles: {len(route_traffic)}")

        safe_tick(world)

        print("[INFO] Spawning cross traffic near junctions...")

        cross_traffic = spawn_cross_traffic(
            world,
            bp_lib,
            world_map,
            spawn_points,
            route,
            traffic_manager,
            protected_actors=[lead, ego] + route_traffic,
            count=CROSS_TRAFFIC_CARS,
        )

        actors.extend(cross_traffic)
        print(f"[INFO] Cross traffic vehicles: {len(cross_traffic)}")

        safe_tick(world)

        print("[INFO] Spawning pedestrians near route...")

        walkers, controllers = spawn_pedestrians_near_route(
            world,
            bp_lib,
            route,
            n=PEDESTRIANS,
        )

        actors.extend(controllers)
        actors.extend(walkers)

        print(f"[INFO] Pedestrians: {len(walkers)}")

        safe_tick(world)

        spectator = world.get_spectator()

        lead_speed_controller = SpeedController(FIXED_DELTA_SECONDS)
        ego_speed_controller = SpeedController(FIXED_DELTA_SECONDS)

        distance_pid = PIDController(
            kp=0.82,
            ki=0.022,
            kd=0.17,
            dt=FIXED_DELTA_SECONDS,
            integral_limit=15.0,
        )

        route_state = {
            "lead_index": 0,
            "ego_index": 0,
        }

        light_wait_memory = {
            "lead": None,
            "ego": None,
            "lead_red_start": None,
            "ego_red_start": None,
            "lead_ignore_light_until": 0.0,
            "ego_ignore_light_until": 0.0,
        }

        final_location = route[-1].transform.location

        print()
        print("=" * 110)
        print("UZB 438E - Real Traffic Car Following Scenario")
        print("Main task: Ego vehicle follows lead vehicle using PID distance control.")
        print("Traffic: NPC vehicles are spawned on the route and near junctions.")
        print("Traffic light recovery: enabled.")
        print("Lead waits for ego if ego is left behind.")
        print(f"Map: {world_map.name}")
        print(f"Desired distance: {DESIRED_DISTANCE_M:.1f} m")
        print(f"Route length: {route_total_length(route):.1f} m")
        print(f"Junction points on route: {count_junction_points(route)}")
        print(f"Route traffic: {len(route_traffic)}")
        print(f"Cross traffic: {len(cross_traffic)}")
        print(f"Pedestrians: {len(walkers)}")
        print(
            f"Start: x={start_wp.transform.location.x:.1f}, "
            f"y={start_wp.transform.location.y:.1f}"
        )
        print(
            f"Final: x={final_location.x:.1f}, "
            f"y={final_location.y:.1f}"
        )
        print("=" * 110)
        print()

        simulation_time = 0.0
        tick_count = 0

        while simulation_time <= SIM_DURATION_SECONDS:
            lead_signal, lead_target_speed, lead_curve, lead_mode = lead_control(
                lead,
                ego,
                route,
                route_state,
                lead_speed_controller,
                world,
                world_map,
                walkers,
                simulation_time,
                light_wait_memory,
            )

            lead.apply_control(lead_signal)

            ego_signal, ego_target_speed, gap, ego_mode = ego_control(
                ego,
                lead,
                route,
                route_state,
                distance_pid,
                ego_speed_controller,
                world,
                world_map,
                walkers,
                simulation_time,
                light_wait_memory,
            )

            ego.apply_control(ego_signal)

            ego_transform = ego.get_transform()
            forward = ego_transform.get_forward_vector()

            spectator.set_transform(
                carla.Transform(
                    ego_transform.location + carla.Location(
                        x=-12.0 * forward.x,
                        y=-12.0 * forward.y,
                        z=7.0,
                    ),
                    carla.Rotation(
                        pitch=-18.0,
                        yaw=ego_transform.rotation.yaw,
                    ),
                )
            )

            if tick_count % PRINT_EVERY_N_TICKS == 0:
                print(
                    f"[t={simulation_time:6.1f}s] "
                    f"Gap={gap:5.2f}m | "
                    f"Lead={get_speed_kmh(lead):5.1f}->{lead_target_speed:4.1f} "
                    f"{lead_mode:<18} | "
                    f"Ego={get_speed_kmh(ego):5.1f}->{ego_target_speed:4.1f} "
                    f"{ego_mode:<18} | "
                    f"Thr={ego_signal.throttle:.2f} "
                    f"Brk={ego_signal.brake:.2f} "
                    f"Str={ego_signal.steer:.2f} | "
                    f"LeadIdx={route_state['lead_index']:3d} "
                    f"EgoIdx={route_state['ego_index']:3d}"
                )

            lead_remaining = len(route) - 1 - route_state["lead_index"]
            ego_remaining = len(route) - 1 - route_state["ego_index"]

            if lead_remaining < 8 and ego_remaining < 25:
                print("[INFO] Final route area reached successfully.")
                break

            if collision_log:
                print("[INFO] Scenario stopped because collision was detected.")
                break

            safe_tick(world)

            simulation_time += FIXED_DELTA_SECONDS
            tick_count += 1

        lead.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                brake=1.0,
                steer=0.0,
            )
        )

        ego.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                brake=1.0,
                steer=0.0,
            )
        )

        safe_tick(world)

        print()

        if collision_log:
            print("[RESULT] Collision detected:")

            for item in collision_log:
                print("  -", item)
        else:
            print("[RESULT] Scenario completed with no detected collision.")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")

    except Exception as exc:
        print("[ERROR]", exc)
        traceback.print_exc()

    finally:
        print("[INFO] Cleaning up...")

        for controller in controllers:
            try:
                controller.stop()
            except Exception:
                pass

        time.sleep(0.3)

        for actor in actors:
            try:
                if hasattr(actor, "stop"):
                    actor.stop()
            except Exception:
                pass

        time.sleep(0.3)

        for actor in actors:
            try:
                actor.destroy()
            except Exception:
                pass

        try:
            if traffic_manager is not None:
                traffic_manager.set_synchronous_mode(False)

            if world is not None and original_settings is not None:
                world.apply_settings(original_settings)

        except Exception:
            pass

        print("[INFO] Done.")


if __name__ == "__main__":
    main()