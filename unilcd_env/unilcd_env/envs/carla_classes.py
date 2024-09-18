#!/usr/bin/env python

"""
This is a custom CARLA/Gymnasium compatible environment meant for training and evaluating the UniLCD router proposed in the paper "Unified Local-Cloud Decision-Making via Residual Reinforcement Learning".

All arguments can be passed as keyword arguments when calling gym.make.
"""


import carla
from carla import ColorConverter as cc
import pygame
import re
import datetime
import numpy as np
import os
import math
import sys
import random
import collections
import weakref
import json
from pathlib import Path

# Pygame key imports
from pygame.locals import KMOD_CTRL, KMOD_SHIFT, K_0, K_9, K_DOWN, K_ESCAPE, K_F1, K_LEFT, K_RIGHT, K_SLASH, K_SPACE, K_TAB, K_UP, K_a, K_c, K_d, K_h, K_q, K_r, K_s, K_w


CONF = {
    "tm_seed": 2021,  # seed for traffic manager
    "car_lights_on": False,
    "percent_walking": 0,  # how many perdestrians will walk
    "percent_crossing": 0,  # how many pedestrians will cross road
    "percent_disabled": 0,
    "max_render_depth_in_meters": 100,
    "min_visible_vertices_for_render": 4,
    "num_vehicles": 200,
    "num_walkers": 30 # Changed from 50 to 0
}


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            # 'Server:  % 16.0f FPS' % self.server_fps,
            # 'Client:  % 16.0f FPS' % clock.get_fps(),
            # '',
            'Agent: % 20s' % get_actor_display_name(
                world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            # 'Simulation time: % 12s' % datetime.timedelta(
            #     seconds=int(self.simulation_time)),
            'Experiment time: %12s' % datetime.timedelta(seconds=int(
                self.simulation_time - world.state_manager.init_t)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 *
                                       math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' %
                                (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' %
                            (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            def distance(l): return math.sqrt((l.x - t.location.x)**2 +
                                              (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x)
                        for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))
        # record

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30)
                                  for x, y in enumerate(item)]
                        pygame.draw.lines(
                            display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255),
                                         rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(
                            display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect(
                                (bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(
                        item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, minimap, state_manager, **kwargs):
        self.world = carla_world
        self.actor_role_name = kwargs['rolename']
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print(
                '  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.minimap = minimap
        self.state_manager = state_manager
        self.player = None
        self.destination_vehicle = None
        self.path_dir = kwargs['path']
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = kwargs['filter']
        self._gamma = kwargs['gamma']
        self.start_id = kwargs['start_id']
        self.destination_id = kwargs['destination_id']
        self.restart()
        self.world.on_tick(self.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False

        self.path_points = []
        self.count = 0
        #
        self.after_init()

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        #spawn_x, spawn_y, spawn_z = -30, 57, 1.5
        # spawn_points = self.map.get_spawn_points()
        # spawn_point = spawn_points[self.start_id]
        spawn_x, spawn_y, spawn_z = self.minimap.planner.path_pts[self.destination_id][0],self.minimap.planner.path_pts[self.destination_id][1],1.5
        #spawn_x, spawn_y, spawn_z = -41.58, -36.86, 1.5
        # Keep same camera config if the camera manager exists.
        #cam_index=0
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        #cam_index=1
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(
            self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(
                blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(
                blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(
                blueprint.get_attribute('speed').recommended_values[2])
        else:
            print("No recommended values for 'speed' attribute")
        # Spawn the player.]
        print("Spawning Player")
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z = spawn_z
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            spawn_point.location.x = spawn_x #-8
            spawn_point.location.y = spawn_y
            spawn_point.rotation.yaw=180.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            # spawn_point = random.choice(
            #     spawn_points) if spawn_points else carla.Transform()\
            spawn_point = spawn_points[self.destination_id]
            # print('location0: ', spawn_point.location.x, spawn_point.location.y, spawn_point.location.z)

            # spawn_point.location.x = 13 #32.6081
            # spawn_point.location.y = 59 #39.0366

            # -7.966956615447998 66.28325653076172
            # spawn_point.location.x = -15 #-8
            # spawn_point.location.y = 59

            # [spawn_x,spawn_y]=random.choice([[-28,57],[-29,56],[-29.8,57.1],[-30, 56],[-31, 56.5],[-31,55]])
            # spawn_point.location.x = spawn_x #-8
            # spawn_point.location.y = spawn_y
            # spawn_point.location.z = spawn_z

            spawn_point.location.x=random.choice([self.minimap.planner.path_pts[self.destination_id][0]-2,self.minimap.planner.path_pts[self.destination_id][0]-1,self.minimap.planner.path_pts[self.destination_id][0],self.minimap.planner.path_pts[self.destination_id][0]+1,self.minimap.planner.path_pts[self.destination_id][0]+2])
            spawn_point.location.y =random.choice([self.minimap.planner.path_pts[self.destination_id][1]-1,self.minimap.planner.path_pts[self.destination_id][1],self.minimap.planner.path_pts[self.destination_id][1]+1])


            # spawn_point.location.x = -14.35 #-8
            # spawn_point.location.y = 58.96

            # spawn_point.location.x = -25 #-8
            # spawn_point.location.y = 58

            spawn_point.rotation.yaw = -180

            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        veh_blueprint = random.choice(
            # self.world.get_blueprint_library().filter('cybertruck'))
            self.world.get_blueprint_library().filter('model3'))

        veh_blueprint.set_attribute('color', '128,128,128')

        while self.destination_vehicle is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = spawn_points[self.start_id]


            # spawn_point.location.x = -41.668877
            # spawn_point.location.x = -38.668877
            # spawn_point.location.y = 48.905540
            # spawn_point.location.z = 0.6
            # spawn_point.rotation.yaw = -90.161217


            self.destination_vehicle = self.world.try_spawn_actor(
                veh_blueprint, spawn_point)
            print("Destination Vehicle spawned at: ", spawn_point)

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(
            self.player, self.hud, self.state_manager)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        self.minimap.tick(self)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        # Run during world loop
        self.hud.tick(self, clock)
        self.minimap.tick(self)
        #print("Hi")

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)
        self.minimap.render(display)
        # pass

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        self.before_destroy()
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()
        if self.destination_vehicle is not None:
            try:
                self.destination_vehicle.destroy()
            except RuntimeError:
                pass

    def after_init(self):
        # Run after world initialization
        print('After initialization')
        return

    def before_destroy(self):
        # Run before world destroyed
        print('Before destroy')
        self.state_manager.save_log()
        return

    def on_world_tick(self, timestamp):
        self.hud.on_world_tick(timestamp)
        self.state_manager.on_world_tick(timestamp)
        

# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""

    def __init__(self, world):
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                    world.minimap.tick(world)
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.constant_velocity_enabled:
                        world.player.disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud.notification(
                            "Disabled Constant Velocity Mode")
                    else:
                        world.player.enable_constant_velocity(
                            carla.Vector3D(17, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud.notification(
                            "Enabled Constant Velocity Mode at 60 km/h")
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")

        # if not self._autopilot_enabled:
        if isinstance(self._control, carla.WalkerControl):
            self._parse_walker_keys(
                pygame.key.get_pressed(), clock.get_time(), world)
        else:
            raise ('Invalid Controller')
        world.player.apply_control(self._control)

    def _parse_walker_keys(self, keys, milliseconds, world):
        print("Enter Keyboard input")
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            # self._rotation.yaw -= 0.04 * milliseconds
            self._rotation.yaw -= 0.005 * milliseconds

        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            # self._rotation.yaw += 0.04 * milliseconds
            self._rotation.yaw += 0.005 * milliseconds


        if keys[K_UP] or keys[K_w]:
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods(
            ) & KMOD_SHIFT else world.player_max_speed


        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- MiniMap ----------------------------------------------------------------
# ==============================================================================

class MiniMap(object):
    def __init__(self, path_dir):
        self.planner = Planner(path_dir)
        return

    def tick(self, world):
        # player info
        self.player_pos = world.player.get_transform().location
        # Camera info
        camera_width = float(
            world.camera_manager.sensor.attributes['image_size_x'])
        camera_height = float(
            world.camera_manager.sensor.attributes['image_size_y'])
        camera_fov = float(world.camera_manager.sensor.attributes['fov'])
        focal = camera_width/(2*np.tan(camera_fov * np.pi / 360))

        world_points = self.planner.path_pts[:, :2]
        path_pts = self._world_to_camera(
            world.camera_manager.sensor, world_points, 1)
        # filter out points with negative x coord (behind camera)
        path_pts = path_pts[:, path_pts[0, :] > 0]
        # apply transformation
        points = self._camera_to_image(camera_width, camera_fov, path_pts)
        # # convert to pygame coord
        self.points = self._image_to_screen(
            camera_width, camera_height, points)

        self.rects = []
        self.ego_rect = None
        marker_size = 0.1
        ego_size = 4
        delta = marker_size*focal / path_pts[0, :]
        # for idx, pt in enumerate(self.points):
        #     self.rects += [pygame.Rect(pt[0]-delta[idx]/2,
        #                                pt[1]-delta[idx]/2, delta[idx], delta[idx])]

        # if camera is topdown camera
        self._points_vis = []
        if world.camera_manager.transform_index == 0:

            self._points_vis = self.points

            for idx, pt in enumerate(self.points):
                self.rects += [pygame.Rect(pt[0]-delta[idx]/2,
                                           pt[1]-delta[idx]/2, delta[idx], delta[idx])]

            ego_world_pos = np.array([[self.player_pos.x, self.player_pos.y]])
            #print("Inside rh",ego_world_pos)
            ego_pt = self._world_to_camera(
                world.camera_manager.sensor, ego_world_pos, 1)
            ego_pt = self._camera_to_image(camera_width, camera_fov, ego_pt)
            ego_pt = self._image_to_screen(camera_width, camera_height, ego_pt)
            self.ego_rect = pygame.Rect(
                ego_pt[0][0]-ego_size/2, ego_pt[0][1]-ego_size/2, ego_size, ego_size)

        # temp: visualizing local goal
        self.goal_rect = None

        local_goal = self.planner.get_next_goal(
            pos=[self.player_pos.x, self.player_pos.y], preview_s=5)
        local_goal = np.array(local_goal)[None, :]
        #print("Inside rh",local_goal)
        local_goal = self._world_to_camera(
            world.camera_manager.sensor, local_goal, 1)
        local_goal = self._camera_to_image(
            camera_width, camera_fov, local_goal)
        local_goal = self._image_to_screen(
            camera_width, camera_height, local_goal)
        #print("Local Goal inside rh", local_goal)
        #print("Ego Point inside rh",ego_pt)
        self.goal_rect = pygame.Rect(
            local_goal[0][0]-ego_size/2, local_goal[0][1]-ego_size/2, ego_size, ego_size)
        return

    def _world_to_camera(self, camera, coord, z_offset):
        # input: Nx3 matrix, [[x1, y1, z1], [x2, y2, z1], ...]
        # output: 3xN matrix [[x1, x2, ...], [y1, y2, ...], [z1, z2, ...]]
        camera_pos = camera.get_transform()
        camera_world_pos = np.array(camera_pos.get_matrix())
        world_camera_pos = np.linalg.inv(camera_world_pos)

        coord = coord.swapaxes(0, 1)
        coord = np.vstack([coord, np.zeros(coord.shape[1])+z_offset,
                          np.ones(coord.shape[1])])  # adding height = 1
        return world_camera_pos @ coord

    def _camera_to_image(self, width, fov, coord):
        # input: 3xN matrix [[x1, x2, ...], [y1, y2, ...], [z1, z2, ...]]
        # output: 3xN matrix [[x1, x2, ...], [y1, y2, ...], [z1, z2, ...]]
        focal = width/(2*np.tan(fov * np.pi / 360))
        reg_coord = coord[:3, :] / coord[0, :]
        mat = np.ones((3, 3))
        mat[1, 1] = mat[2, 2] = focal
        return mat @ reg_coord

    def _image_to_screen(self, width, height, coord):
        # input: 3xN matrix [[x1, x2, ...], [y1, y2, ...], [z1, z2, ...]]
        # output: Nx2 matrix [[x1, y1], [x2, y2], ...]
        coord = coord.swapaxes(0, 1)
        pts = np.zeros((coord.shape[0], 2))
        pts[:, 0] = width/2 + coord[:, 1]
        pts[:, 1] = height/2 - coord[:, 2]
        return pts

    def render(self, display):
        # if len(self.points) > 1:
        if len(self._points_vis) > 1:
            pygame.draw.lines(display, (255, 0, 0), False, self._points_vis)

            pygame.draw.rect(display, (0, 0, 255), self.goal_rect)

        for rect in self.rects:
            pygame.draw.rect(display, (255, 0, 0), rect)

        if self.ego_rect:
            pygame.draw.rect(display, (0, 255, 0), self.ego_rect)

        # if self.goal_rect:
        #     pygame.draw.rect(display, (0, 0, 255), self.goal_rect)
        return


class Planner():
    def __init__(self, path_file):
        self.path_file = path_file
        data = np.flip(np.load(path_file, allow_pickle=True)[:],axis=0)
        # print(data)
        s = np.add.accumulate(np.linalg.norm(np.diff(data[:,:2], axis = 0), axis = 1))
        s = np.concatenate(([0], s))
        x = data[:,0:1]
        y = data[:,1:2]
        self.path_pts = np.concatenate((x, y, s[:,None]), axis=1) # x, y, s
        self.path_length = len(self.path_pts)
        self.path_length_value=0
        return
    

    def get_next_goal(self, pos, preview_s):
        pos = np.array(pos)
        closest_pt_id = self.get_closest_point(pos)
        # print('closest: ',closest_pt_id, pos)
        for i in range(closest_pt_id, self.path_length):
            ds = self.path_pts[i, 2] - self.path_pts[closest_pt_id, 2]
            if ds >= preview_s:
                # print('point goound: ', i, ds)
                return self.path_pts[i, [0,1]]
        # return final point if not found
        return self.path_pts[-1, 0], self.path_pts[-1, 1]
    

    def get_closest_point(self, pos):
        return np.argmin(np.linalg.norm(self.path_pts[:,0:2] - pos,  axis = 1))

    def get_path_length(self):
        for i in range(1,self.path_length):
            self.path_length_value+=np.linalg.norm(self.path_pts[i]-self.path_pts[i-1])
        return self.path_length_value

    def get_path_length_from_position(self, pos, dest):
        # print(self.path_pts.shape)
        # print(pos.shape)
        index = np.argwhere((self.path_pts[:,0:2]==pos).all(1)==True)[0][0]
        dest_index = -1
        # print("Indices:",index, dest_index)
        # path_length_value = []
        # for i in range(index+1, self.path_length):
        #     path_length_value.append(np.linalg.norm(self.path_pts[i,0:2]-self.path_pts[i-1,0:2]))
        return self.path_pts[dest_index,2]-self.path_pts[index,2]


# ==============================================================================
# -- StateManager --------------------------------------------------------------
# ==============================================================================

# handle collision
# handle timeout
# handle goal reached
# log info


class StateManager(object):
    def __init__(self, minimap, log_dir):
        self.log = {'config': CONF, 'event': []}
        self.log_dir = Path(log_dir)
        self.end_state = False
        self.start_state = False
        self.time_limit_seconds = 100000000000
        self.simulation_time = 0
        self.init_t = 0
        self.last_collision_t = 0
        self.last_collision_type = None
        self.minimap = minimap
        return

    def on_world_tick(self, timestamp):
        self.simulation_time = timestamp.elapsed_seconds
        self.timeout_handler(self.simulation_time)
        if self.start_state:
            self.success_handler(self.simulation_time)
        return

    def start_handler(self, timestamp):
        self.start_state = True
        self.end_state = False
        self.init_t = self.simulation_time
        self.log['path_dir'] = self.minimap.planner.path_file
        self.log['event'].append(self._event(0,  'Start experiment'))
        print(self.log)
        return

    def success_handler(self, timestamp):
        radius = 1.3 # reached goal if distance < radius
        player_pos = np.array([self.minimap.player_pos.x,
                              self.minimap.player_pos.y])
        goal = self.minimap.planner.path_pts[-1, 0:2]
        d = np.sqrt(np.sum((player_pos - goal)**2))
        if d < radius:
            # success
            delta_t = timestamp - self.init_t
            self.log['event'].append(self._event(delta_t, "Goal reached"))
            self._end(success=True, info=self._event(delta_t, "Goal reached"))
        return

    def collision_handler(self, actor_type):
        # TODO: handle walker and vehicle colliison
        freq_thresh = 1.0
        delta_t = self.simulation_time - self.init_t
        if actor_type == 'Sidewalk':
            return

        if delta_t - self.last_collision_t > freq_thresh or actor_type != self.last_collision_type:
            self.log['event'].append(self._event(
                delta_t, 'Collide with %s' % actor_type))
            self.last_collision_t = delta_t
            self.last_collision_type = actor_type
            print(delta_t, actor_type)
            if actor_type == 'Vehicle':
                self._end(success=False, info=self._event(
                    delta_t, 'Collide with %s' % actor_type))
        return

    def timeout_handler(self, timestamp):
        if self.start_state:
            delta_t = timestamp - self.init_t
            if delta_t > self.time_limit_seconds:
                self.log['event'].append(self._event(delta_t, "Time out"))
                self._end(success=False, info=self._event(delta_t, "Time out"))
                print(self.log)
        return

    def _end(self, success, info):
        self.log['result'] = {'success': success, 'info': info}
        self.start_state = False
        self.end_state = True

    @staticmethod
    def _event(timestamp, event_text):
        return {'timestamp': timestamp, 'info': event_text}

    def save_log(self):
        if not os.path.exists(str(self.log_dir)):
            self.log_dir.mkdir(parents=True, exist_ok=True)
        time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        save_path = self.log_dir/(time + '_log.json')
        with open(save_path, 'w') as f:
            json.dump(self.log, f)
            print('Log saved to: ',save_path)
        return


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud, state_manager):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        self.state_manager = state_manager
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity, actor_id in self.history:
            history[frame] += intensity
        return history

    # @staticmethod
    # def _on_collision(weak_self, event):
    #     self = weak_self()
    #     if not self:
    #         return
    #     actor_type = get_actor_display_name(event.other_actor)
    #     self.hud.notification('Collision with %r' % actor_type)
    #     self.state_manager.collision_handler(actor_type)
    #     # print('Collision with %r' % actor_type)
    #     impulse = event.normal_impulse
    #     intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
    #     self.history.append((event.frame, intensity))
    #     if len(self.history) > 4000:
    #         self.history.pop(0)

    # Modified versions of above code for resetting based on collision with walkers and vehicles
    def get_collision_ids(self):
        history = collections.defaultdict(str)

        for frame, intensity, actor_id in self.history:
            history[frame] += actor_id
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        self.state_manager.collision_handler(actor_type)
        # print('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity, event.other_actor.type_id))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(
            carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(
                sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(
                sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)
    

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        Attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=0.0, z=50.0), carla.Rotation(pitch=-90)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=1., z=0.)), Attachment.Rigid),
            ]
        self.transform_index = 1
        self.sensors = [
            # ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {'fov': '110'}]]
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}]]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index +
                                1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index]
             [2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' %
                              ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]
                    ['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(
                dvs_img.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)
        

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""

    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 *
                    self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)
            

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters)
               if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_matrix(transform):
    """
    Creates matrix from carla transform.
    """

    rotation = transform.rotation
    location = transform.location
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    c_r = np.cos(np.radians(rotation.roll))
    s_r = np.sin(np.radians(rotation.roll))
    c_p = np.cos(np.radians(rotation.pitch))
    s_p = np.sin(np.radians(rotation.pitch))
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix