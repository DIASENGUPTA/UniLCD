import carla
import numpy as np
import random
import os, sys
import glob
import queue

# Setting up libcarla import by adding egg file to system path (may not be required)
try:
    sys.path.append(glob.glob('./unilcd_env/envs/carla-*%d.%d-%s.egg' % (
    sys.version_info.major,
    sys.version_info.minor,
    'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

from libcarla.command import SpawnActor as SpawnActor
from libcarla.command import SetAutopilot as SetAutopilot
from libcarla.command import SetVehicleLightState as SetVehicleLightState
from libcarla.command import FutureActor as FutureActor


# Constants

WEATHER = {
    'ClearNoon':carla.WeatherParameters.ClearNoon,
    'ClearSunset': carla.WeatherParameters.ClearSunset,
    'WetNoon': carla.WeatherParameters.WetNoon,
    'HardRainNoon': carla.WeatherParameters.HardRainNoon,
    'WetCloudyNoon': carla.WeatherParameters.WetCloudyNoon,
    'SoftRainSunset': carla.WeatherParameters.SoftRainSunset,
    'HardRainSunset': carla.WeatherParameters.HardRainSunset,
    'WetCloudySunset': carla.WeatherParameters.WetCloudySunset
}

regularped_ids = ['0001','0002','0003','0004','0005',
                  '0006','0007','0008','0009','0010',
                  '0011','0012','0013','0014','0015',
                  '0016','0017','0018','0019','0020','0021'
                  ]

def get_image(image, _type=None):
    if _type == 'semantic_segmentation':
        image.convert(carla.ColorConverter.CityScapesPalette)

    _array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    _array = np.reshape(_array, (image.height, image.width, 4))
    _array = _array[:, :, :3]
    _array = _array[:, :, ::-1]
    
    return _array


def spawn_cameras(world, walker, destination_vehicle, width, height, fov):
    cameras = {}
    actor = walker
    # eye-level camera
    x_forward_eyelevel = 1  # 0.4
    cam_height_eyelevel = 0.0 #0.8  # 0.5

    fov = 120
    w = 1024# 1280
    h = 576 # 720

    cameras['eyelevel_rgb'] = spawn_camera(world, actor, 'rgb', w, h, fov,
                                           x_forward_eyelevel, 0.0, cam_height_eyelevel, 0.0, 0.0)

    cameras['eyelevel_ins'] = spawn_camera(world, actor, 'instance_segmentation', w, h, fov,
                                           x_forward_eyelevel, 0.0, cam_height_eyelevel, 0.0, 0.0)


    cameras['eyelevel_dep'] = spawn_camera(world, actor, 'depth', w, h, fov,
                                            x_forward_eyelevel, 0.0, cam_height_eyelevel, 0.0, 0.0)

    return cameras

def spawn_camera(world, vehicle, _type, w, h, fov, x, y, z, pitch, yaw):
    if _type == 'rgb':
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    elif _type == 'instance_segmentation':
        camera_bp = world.get_blueprint_library().find(
            'sensor.camera.instance_segmentation')
    elif _type == 'semantic_segmentation':
        camera_bp = world.get_blueprint_library().find(
            'sensor.camera.semantic_segmentation')
    elif _type == 'depth':
        camera_bp = world.get_blueprint_library().find('sensor.camera.depth')

    camera_bp.set_attribute('image_size_x', str(w))
    camera_bp.set_attribute('image_size_y', str(h))
    camera_bp.set_attribute('fov', str(fov))
    _camera = world.spawn_actor(
        camera_bp,
        carla.Transform(carla.Location(x=x, y=y, z=z),
                        carla.Rotation(pitch=pitch, yaw=yaw)),
        attach_to=vehicle)

    view_width = w
    view_height = h
    view_fov = fov
    if _type == 'rgb':
        calibration = np.identity(3)
        calibration[0, 2] = view_width / 2.0
        calibration[1, 2] = view_height / 2.0
        calibration[0, 0] = calibration[1, 1] = view_width / \
            (2.0 * np.tan(view_fov * np.pi / 360.0))
        _camera.calibration = calibration

    return _camera


def spawn_walkers(client, world, N, percent_disabled, percent_walking, percent_crossing, all_id_dict):
    spawn_pts = []


    for i in range(N):
        spawn_pt = carla.Transform()
        random_loc = world.get_random_location_from_navigation()
        #random_loc.z=0.5999999642372131
        if random_loc != None:
            spawn_pt.location = random_loc
            spawn_pts.append(spawn_pt)
    print('walker spawn pts: ', len(spawn_pts))

    batch = []
    walkers_list = []
    walker_speed = []
    for pt in spawn_pts:
        # walker_bp = random.choice(world.get_blueprint_library().filter('walker'))

        # if (random.random() < 0.1):
        #     if (random.random() < 0.2):
        #         disabled_id = random.choice(visuallyimpairedped_all_ids)
        #     else:
        #         disabled_id = random.choice(wheelchairped_all_ids)
        #     walker_bp = world.get_blueprint_library().filter('walker.pedestrian.'+disabled_id)[0]
        #     if_disabled = True
        #     #print("disable id ", disabled_id)
        # else:
        #     regular_id = random.choice(regularped_ids)
        #     walker_bp = world.get_blueprint_library().filter('walker.pedestrian.'+regular_id)[0]
        #     #print("regular id ", regular_id)

        regular_id = random.choice(regularped_ids)
        walker_bp = world.get_blueprint_library().filter('walker.pedestrian.'+regular_id)[0]


        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        if walker_bp.has_attribute('speed'):
            if (random.random() < percent_walking):
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
            else:
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
        else:
            print("Walker has no speed")
            walker_speed.append(0.0)
        # print(batch)
        batch.append(SpawnActor(walker_bp, pt))
    
    count = 0
    for response in client.apply_batch_sync(batch, True):
        if response.error:
            count +=1
            # print(response.error)
        else:
            walkers_list.append(response.actor_id)

    print("Number of sparse walkers failed to spawn: ",count)

    batch = []
    controller_list = []
    controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for walker in walkers_list:
        batch.append(SpawnActor(controller_bp, carla.Transform(), walker))
    for response in client.apply_batch_sync(batch, True):
        if response.error:
            print(response.error)
        else:
            controller_list.append(response.actor_id)

    all_id_dict.update({'walkers': walkers_list})
    all_id_dict.update({'controllers': controller_list})

    world.tick()

    world.set_pedestrians_cross_factor(percent_crossing)

    locations_list = []
    for i in range(len(walkers_list)):
        start_location = world.get_actor(walkers_list[i]).get_transform()
        #start_location.z=0.5999999642372131
        controller = world.get_actor(controller_list[i])
        controller.start()
        m=world.get_random_location_from_navigation()
        #m.z=0.5999999642372131
        locations_list.append((start_location,m))
        controller.go_to_location(m)

    all_id_dict.update({'locations': locations_list})
    
    # for id in controller_list:
    #     controller = world.get_actor(id)
    #     controller.start()
    #     m=world.get_random_location_from_navigation()
    #     #m.z=0.5999999642372131
    #     controller.go_to_location(m)
    #     # all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

    # for walker in walkers_list:
    #     print(world.get_actor(walker).get_velocity())
    return True



def sample_around(loc, radius = 50, N = 100):
    #z = loc[2] # fixed spawn height
    z=0.5999999642372131
    xs = np.random.uniform(loc[0]-radius, loc[0]+radius, N)
    ys = np.random.uniform(loc[1]-radius, loc[1]+radius, N)
    coord = np.stack((xs, ys, [z]*N), axis = 1)
    return coord


def spawn_walkers_dense(client, world, N, loc_center, percent_disabled, percent_walking, percent_crossing, all_id_dict, destination_pt=None):
    spawn_pts = []

    # sampled_points=np.load('final_vid_new.npy')
    # sampled_points=np.load('data_final.npy')
    sampled_points = sample_around(loc_center, radius = 20, N=N)
    # np.save('data_14_11.npy', sampled_points)
    # destination_loc = [destination_pt[0],destination_pt[1], 0.5999999642372131]
    # midpt_loc = [(a+b)/2 for a,b in zip(loc_center,destination_loc)]
    # midpt_radius = int(np.round(np.linalg.norm(np.array(loc_center) - np.array(destination_loc)) * 0.5))
    #N = 70 # N if midpt_radius < N/2 else

    # sampled_points = sample_around(midpt_loc, radius = midpt_radius, N=N)
    # np.save('data_27_11.npy', sampled_points)
    for i in range(N):
        pt = sampled_points[i, :]
        spawn_point = carla.Transform()
        loc = carla.Location(x = pt[0], y = pt[1], z = pt[2])
        spawn_point.location = loc
        spawn_pts.append(spawn_point)


    # for i in range(N):
    #     spawn_pt = carla.Transform()
    #     random_loc = world.get_random_location_from_navigation()
    #     if random_loc != None:
    #         spawn_pt.location = random_loc
    #         spawn_pts.append(spawn_pt)
    print('walker spawn pts: ', len(spawn_pts))

    batch = []
    walkers_list = []
    walker_speed = []
    
    for pt in spawn_pts:
        # walker_bp = random.choice(world.get_blueprint_library().filter('walker'))

        # if (random.random() < 0.1):
        #     if (random.random() < 0.2):
        #         disabled_id = random.choice(visuallyimpairedped_all_ids)
        #     else:
        #         disabled_id = random.choice(wheelchairped_all_ids)
        #     walker_bp = world.get_blueprint_library().filter('walker.pedestrian.'+disabled_id)[0]
        #     if_disabled = True
        #     #print("disable id ", disabled_id)
        # else:
        #     regular_id = random.choice(regularped_ids)
        #     walker_bp = world.get_blueprint_library().filter('walker.pedestrian.'+regular_id)[0]
        #     #print("regular id ", regular_id)

        regular_id = random.choice(regularped_ids)
        walker_bp = world.get_blueprint_library().filter('walker.pedestrian.'+regular_id)[0]

        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        if walker_bp.has_attribute('speed'):
            if (random.random() < percent_walking):
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
            else:
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
        else:
            print("Walker has no speed")
            walker_speed.append(0.0)
        batch.append(SpawnActor(walker_bp, pt))
    ct=0
    for response in client.apply_batch_sync(batch, False):
        if response.error:
            ct+=1
            # print(response.error)
        else:
            walkers_list.append(response.actor_id)

    batch = []
    controller_list = []
    controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for walker in walkers_list:
        batch.append(SpawnActor(controller_bp, carla.Transform(), walker))
    
    for response in client.apply_batch_sync(batch, False):
        if response.error:
            print(response.error)
        else:
            controller_list.append(response.actor_id)

    print("Number of pedestrians spawned:",len(spawn_pts)-ct)
    all_id_dict.update({'walkers': walkers_list})
    all_id_dict.update({'controllers': controller_list})

    world.tick()

    world.set_pedestrians_cross_factor(percent_crossing)

    locations_list=[]
    for i in range(len(walkers_list)):
        start_location = world.get_actor(walkers_list[i]).get_transform()
        #start_location.z=0.5999999642372131
        controller = world.get_actor(controller_list[i])
        controller.start()
        m=world.get_random_location_from_navigation()
        m.z=0.5999999642372131
        locations_list.append((start_location,m))
        controller.go_to_location(m)
        #location=[]
        # location.append(m.x)
        # location.append(m.y)
        # location.append(m.z)
        # location_reach.append(location)
        

    
    # carla_location_serialized = pickle.dumps(location_reach)
    # with open('locations.pkl', 'wb') as file:
    #     file.write(carla_location_serialized)
    #np.save('data1.npy', location_reach)

    all_id_dict.update({'locations': locations_list})
    
    # for id in controller_list:
    #     controller = world.get_actor(id)
    #     controller.start()
    #     m=world.get_random_location_from_navigation()
    #     m.z=0.5999999642372131
    #     controller.go_to_location(m)
        # all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

    # for walker in walkers_list:
    #     print(world.get_actor(walker).get_velocity())
    return True