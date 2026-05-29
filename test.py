"""Lead vehicle autopilot test - Town01 (lighter map)"""
import carla, time, math, random

def get_speed(v):
    vel = v.get_velocity()
    return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6

client = carla.Client("localhost", 2000)
client.set_timeout(120.0)
print("Connected:", client.get_server_version())

print("Loading Town01...")
world = client.load_world("Town01")
print("Loaded!")
time.sleep(3.0)

bp = world.get_blueprint_library().find("vehicle.tesla.model3")
sp = world.get_map().get_spawn_points()
random.shuffle(sp)

vehicle = world.spawn_actor(bp, sp[0])
print("Spawned at:", sp[0].location)
time.sleep(2.0)

tm = client.get_trafficmanager(8000)
tm.set_synchronous_mode(False)
vehicle.set_autopilot(True, 8000)
print("Autopilot ON")

for i in range(60):
    time.sleep(0.5)
    spd = get_speed(vehicle)
    print(f"[{i*0.5:.1f}s] Speed: {spd:.1f} km/h")
    if spd > 1.0:
        print("SUCCESS: Vehicle is moving!")
        break

vehicle.destroy()
print("Done.")