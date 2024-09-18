import carla
import numpy as np
import pygame
import logging
from . import util

# Import CARLA classes
from .carla_classes import HUD, World, MiniMap, StateManager, KeyboardControl


def start_carla_connection(
    # client_timeout: float = 60.0,
    **kwargs,
):
    display = pygame.display.set_mode(
        (kwargs['width'], kwargs['height']),
        pygame.HWSURFACE | pygame.DOUBLEBUF
    )
    world = None
    vehicles_list = []
    log_level = logging.debug if kwargs['debug'] else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    
    
    # Get carla client
    client, world, minimap, state_manager, traffic_manager, controller = initCARLAObjs(**kwargs)
    
    sparse_walkerId_dict, dense_walkerId_dict = spawnWalkers(client=client, world=world, minimap=minimap)
    
    return client, world, minimap, state_manager, traffic_manager, controller, display, sparse_walkerId_dict, dense_walkerId_dict


def initCARLAObjs(**kwargs):
    pygame.init()
    # Start CARLA Client
    client = carla.Client(kwargs['host'],kwargs['port'])
    client.set_timeout(kwargs['client_timeout'])
    
    # Load HUD, Minimap, State Manager
    hud = HUD(kwargs['width'], kwargs['height'])
    minimap = MiniMap(kwargs['path'])
    state_manager = StateManager(minimap, kwargs['log_dir'])
    
    # Loading onto client
    client.load_world(kwargs['world'])
    client.get_world().unload_map_layer(carla.MapLayer.ParkedVehicles)
    client.get_world().unload_map_layer(carla.MapLayer.Props)
    client.get_world().unload_map_layer(carla.MapLayer.Decals)
    
    # Load world
    world = World(client.get_world(), hud, minimap, state_manager, **kwargs)
    settings = world.world.get_settings()
    synchronous_master = True
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    
    world.world.apply_settings(settings)
    logging.info("World Loaded")
    
    if kwargs['weather'] in util.WEATHER:
        weather = util.WEATHER[kwargs['weather']]
    else:
        weather = util.WEATHER['ClearNoon']
    
    world.world.set_weather(weather)
    logging.info("Weather Initialized")
    
    # Start traffic manager and init settings
    traffic_manager = client.get_trafficmanager(kwargs['tm_port'])
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_hybrid_physics_mode(True)
    traffic_manager.set_hybrid_physics_radius(70.0)
    traffic_manager.set_synchronous_mode(True)
    
    # Get world settings
    settings = world.world.get_settings()
    if not settings.synchronous_mode:
        synchronous_master = True
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
    else:
        synchronous_master = False
    
    logging.info(f"Synchronous:  {synchronous_master}")
    
    controller = KeyboardControl(world)
    
    return client, world, minimap, state_manager, traffic_manager, controller    

def spawnWalkers(**kwargs):
    # Spawn walkers sparsely in world
    sparse_id_dict = {}
    if not util.spawn_walkers(kwargs['client'], kwargs['world'].world, 30, 0, 0.9, 0.2, sparse_id_dict):
        logging.info("Walkers failed to spawn (sparse setting)")
        return
    
    player_location = kwargs['world'].player.get_transform()
    player_loc_coords = [player_location.location.x, player_location.location.y, player_location.location.z]
    
    destination_coords = np.array([
        kwargs['minimap'].planner.path_pts[-1][0], kwargs['minimap'].planner.path_pts[-1][1] 
    ])
    
    # Spawn walkers densely (around ego robot and destination)
    dense_id_dict = {}
    if not util.spawn_walkers_dense(kwargs['client'], kwargs['world'].world, 30, player_loc_coords, 0, 0.9, 0.2, dense_id_dict, destination_coords):
        logging.info("Walkers failed to spawn (dense setting)")
        return
    
    all_actors = kwargs['world'].world.get_actors()
    all_vehicles = []
    all_peds = []
    for _a in all_actors:
        if 'vehicle' in _a.type_id:
            # print(_a.type_id)
            all_vehicles.append(_a)

        if  _a.type_id.startswith('walker'):
            # print(_a.type_id)
            all_peds.append(_a)
            
    return sparse_id_dict, dense_id_dict
    
    
    
    
    
    
    
    