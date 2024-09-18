import gymnasium as gym
from .env_setup import start_carla_connection
import pygame
import numpy as np
import torch
import torchvision.transforms as transforms
import time
import carla
import cv2
from gymnasium import spaces
from . import util
import math
from .CarlaSyncMode import CarlaSyncMode
import random
import pandas as pd
from .il_models.cloud_model import CloudModel
from .il_models.local_model import LocalModel


"""
This is a custom CARLA/Gymnasium compatible environment meant for training and evaluating the UniLCD router proposed in the paper "Unified Local-Cloud Decision-Making via Residual Reinforcement Learning".

All arguments can be passed as keyword arguments when calling gym.make.
"""


class UniLCDEmbEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'env_mode': ['train', 'eval']}
    def __init__(self, **kwargs):
        self.client, self.world, self.minimap, self.state_manager, self.traffic_manager, self.controller, self.display, self.sparse_walkerId_dict, self.dense_walkerId_dict = start_carla_connection(**kwargs)
        
        self.distance_covered = 0
        
        self.im_width = kwargs['width']
        self.im_height = kwargs['height']
        
        self.steps_per_episode = int(kwargs['steps_per_episode'])
        
        self.clock = pygame.time.Clock()
        
        assert kwargs['render_mode'] is None or kwargs['render_mode'] in self.metadata["render_modes"]
        self.render_mode = kwargs['render_mode']
        self.env_mode = kwargs['env_mode']
        
        
        # Task Metrics for Evaluation
        self.task_metrics = []
        self.action_list = np.array([
            [0, 0, 0]
        ])
        
        self.total_energy_consumption = 0.0
        self.total_latency = 0.0
        
        self.sampled_latency = 0
        
        self.device = torch.device(f"cuda:{kwargs['device']}") if kwargs['device'] else torch.device('cpu')
        
        # Initialize Imitation Learning Models for Cloud and Edge (local) setting (IL models always in eval mode during training/eval of UniLCD)
        # 
        self.cloud_model = CloudModel().load_state_dict(torch.load(kwargs['cloud_model_checkpoint'])).to(self.device).eval()
        self.local_model = LocalModel().load_state_dict(torch.load(kwargs['local_model_checkpoint'])).to(self.device).eval()
        
        self.tensorTransform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.resizeTransform = transforms.Resize(
            (96,96),
            antialias=True
        )
        
        self.normalizeTransform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225],
        )
        
        self.ped_count = 0
        self.num_pedestrian_infractions = 0
        self.prev_collisions = []
        self.curr_collisions = []
        self.pedestrian_list = []
        
        self.episode_start_time = time.time()
        self.episode_step = 0
        self.episode_count = 0
        
        
        self.target_wpt = carla.Location(
            x = self.minimap.planner.path_pts[-1,0:1][0],
            y = self.minimap.planner.path_pts[-1,1:2][0],
            z = 0.0,
        )
        self.curr_wpt_index = 0
        pos = [self.world.destination_vehicle.get_location().x, self.world.destination_vehicle.get_location().y]

        self.start_wpt_index = self.minimap.planner.get_closest_point(pos)
        self.start_wpt = self.minimap.planner.path_pts[self.start_wpt_index,0:2]
        self.dest_wpt_index = self.minimap.planner.get_closest_point(pos)
        self.dest_wpt = self.minimap.planner.path_pts[self.dest_wpt_index, 0:2]

        self.route_length = self.minimap.planner.get_path_length_from_position(self.start_wpt, self.dest_wpt)
        
        self.wpts = self.minimap.planner.path_pts[self.start_wpt_index:]
        self._accum_meters = self.minimap.planner.path_pts[self.start_wpt_index:,2] - self.minimap.planner.path_pts[self.start_wpt_index,2]
        
        # Eval-time variables
        if self.env_mode == 'eval':
            self.rollout_video = cv2.VideoWriter(
                kwargs['rollout_video'], 
                cv2.VideoWriter_fourcc(*'MJPG'), 
                kwargs['fps'], (self.im_width, 
                self.im_height)
            )
            self.minimap_video = cv2.VideoWriter(
                kwargs['minimap_video'], 
                cv2.VideoWriter_fourcc(*'MJPG'), 
                kwargs['fps'], (self.im_width, 
                self.im_height)
            )
            self.task_metrics_path = kwargs['task_metrics_path']
        
        print("Environment initialized")
        
    
    @property
    def observation_space(self, *args, **kwargs):
        return spaces.Box(low=np.array([[-np.inf]*48+[-3.14, 0.0,0.0]]), high=np.array([[np.inf]*48+[3.14, 2.0,1.0]]),dtype=np.float64)

        
    @property
    def action_space(self):
        return spaces.Discrete(2)
    
    def compute_local_policy(self, image, location):
        img_tensor = self.normalizeTransform(self.tensorTransform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))[:,:,80:560]).unsqueeze(0)
        location_tensor = torch.tensor(location, dtype=torch.float32).unsqueeze(0)
        # res=self.local_model(img_tensor.to(self.device), location_tensor.to(self.device))
        res, emb = self.local_model(img_tensor.to(self.device), location_tensor.to(self.device),True)
        res=res.cpu().detach().numpy()
        emb=emb.squeeze(0).cpu().detach().numpy()
        self.action_list = np.vstack((self.action_list, np.array([res[0][0],res[0][1],0.0])))
        energy=0.15
        return self.action_list[-4:],emb,energy
    
    def compute_cloud_policy(self, pedestrian_detected, image, location):
        img_tensor = self.normalizeTransform(self.tensorTransform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))).unsqueeze(0)
        location_tensor = torch.tensor(location, dtype=torch.float32).unsqueeze(0)
        res=self.cloud_model(img_tensor, location_tensor)
        res=res.cpu().detach().numpy()
        
        if pedestrian_detected==0:
            self.speed_new = res[0][1]
            self.rotation_new=res[0][0]
        else:
            self.speed_new = 0
            self.rotation_new=0
        self.action_list[-1][0] = self.rotation_new
        self.action_list[-1][1] = self.speed_new
        self.action_list[-1][2] = 1.0
        
        energy=1.5
        return self.action_list[-1], energy
    
    def initialAct(self):
        self.ped_count = 0
        player_pos = self.world.player.get_location()
        ped_distances = []
        for i in self.pedestrian_list:
            ped_distances.append(self._get_actor_distance(i, player_pos))
            if self._get_actor_distance(i, player_pos) < 4 and self._get_actor_direction(i, player_pos) != None:
                self.ped_count+=1

        pedestrian_detected = 1.0 if self.ped_count > 1 else 0.0
        path_next_x,path_next_y = self.minimap.planner.get_next_goal(pos=[self.world.player.get_transform().location.x,self.world.player.get_transform().location.y],preview_s=5)
        
        wpt =np.array([path_next_x,path_next_y]) - np.array([self.world.player.get_transform().location.x,self.world.player.get_transform().location.y])
        action,embedding,local_energy=self.compute_local_policy(self.eyelevel_rgb_array,wpt)
        
        return action, embedding, pedestrian_detected, local_energy, wpt
        
    def _get_obs(self):
        self.snapshot, self.eyelevel_rgb, self.eyelevel_ins, self.eyelevel_dep = self.sync_mode.tick(None)
        self.eyelevel_rgb_array = cv2.resize(np.uint8(np.array(util.get_image(self.eyelevel_rgb))), (self.im_width, self.im_height))[:,:,:3]
        self.eyelevel_rgb_array = cv2.cvtColor(self.eyelevel_rgb_array, cv2.COLOR_RGB2BGR)
        
        cv2.imshow("Ego Camera View", self.eyelevel_rgb_array)
        cv2.waitKey(1)
        
        self.collision_count = 0
        player_pos = self.world.player.get_location()
        self.curr_collisions = []
        
        for ped in self.pedestrian_list:
            if self._get_actor_distance(ped, player_pos) < 1 and self._get_actor_direction(ped, player_pos) != None:
                self.collision_count +=1
                if ped.id != self.world.player.id:
                    self.curr_collisions.append(ped.id)
        
        collision_detected = 0.0
        if self.collision_count > 1 and (sorted(self.curr_collisions) != self.prev_collisions):
            collision_detected = 1.0
        
        self.num_pedestrian_infractions += int(collision_detected)
        self.prev_collisions = sorted(self.curr_collisions)
        
        closest_wpt = self.minimap.planner.get_closest_point(
            np.array([
                self.world.player.get_transform().location.x,
                self.world.player.get_transform().location.y
            ])
        )
        
        closest_wpt_x, closest_wpt_y = self.minimap.planner.path_pts[closest_wpt, :2]
        
        nxt_wpt = np.array(
            self.minimap.planner.get_next_goal(
                pos=[self.world.player.get_transform().location.x,
                    self.world.player.get_transform().location.y],
                preview_s=5
            )
        )[None, :]
        
        # ego_pos = np.array([[self.world.player.get_transform().location.x, self.world.player.get_transform().location.y]])
        
        angle = math.atan2(
            nxt_wpt[0][1] - player_pos.y, nxt_wpt[0][0] - player_pos.x
        )
        
        distance_covered = np.linalg.norm(
            np.array([closest_wpt_x, closest_wpt_y]) - np.array([player_pos.x, player_pos.y])
        )
        
        if angle >= -1.57 and angle <= 1.57:
            r = 1
        else:
            r = 0
            
        return np.array([collision_detected, distance_covered, r], dtype = np.float64)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1
        self.episode_step = 0
        self.step_distance = 0
        self.distance_covered = 0
        self.num_pedestrian_infractions = 0
        self.total_latency = 0
        self.local_step_counter = 0
        self.cloud_step_counter = 0

        self.total_energy_consumption = 0.0
        self.percentage_route_completion = 0.0
        self.curr_wpt_index = 0
        
        if self.episode_count > 1:
            for camera in list(self.cameras.values()):
                camera.stop()
                camera.destroy()
                
            for i in range(len(self.dense_walkerId_dict)):
                self.world.world.get_actor(self.dense_walkerId_dict['controllers'][i]).stop()
        
            self.action_list = np.array([
                [0, 0, 0]
            ])
            
            self.client.apply_batch([
                carla.command.DestroyActor(x) for x in self.dense_walkerId_dict['controllers']
            ])
            self.world.tick(self.clock)
            
            walker_transforms = [
                carla.command.ApplyWalkerState(actor_id, x[0], 0.0) for actor_id, x in zip(self.dense_walkerId_dict['walkers'], self.dense_walkerId_dict['locations'])
            ]
            self.client.apply_batch(walker_transforms)
            self.world.tick(self.clock)
            
            batch = []
            controller_list = []
            controller_bp = self.world.world.get_blueprint_library().find('controller.ai.walker')
            for walker in self.dense_walkerId_dict['walkers']:
                batch.append(carla.libcarla.command.SpawnActor(controller_bp, carla.Transform(), walker))
            for response in self.client.apply_batch_sync(batch, True):
                if response.error:
                    print(response.error)
                else:
                    controller_list.append(response.actor_id)

            #all_id_dict.update({'walkers': walkers_list})
            self.dense_walkerId_dict.update({'controllers': controller_list})
            
            
            for i in range(len(self.dense_walkerId_dict['controllers'])):
                location = self.dense_walkerId_dict['locations'][i]

                controller = self.world.world.get_actor(self.dense_walkerId_dict['controllers'][i])
                controller.start()
                controller.go_to_location(location[1])
            
            self.world.restart()
        
        self.cameras = {}
        self.cameras = util.spawn_cameras(self.world.world, self.world.player, self.world.destination_vehicle, self.im_width, self.im_width, 90)
        
        self.sync_mode = CarlaSyncMode(self.world.world, *list(self.cameras.values()), fps = 20)
        
        try:
            self.sync_mode.__enter__()
        finally:
            self.sync_mode.__exit__()
        
        self.world.tick(self.clock)
        
        self.pedestrian_list = []
        p_list = self.world.world.get_actors().filter("walker.*")
        v_list = self.world.world.get_actors().filter("vehicles.*")
        self.pedestrian_list = list(p_list)+list(v_list)
        
        obs = self._get_obs()
        
        self.info = dict()
        
        self.state_manager.start_handler(self.world)
        self.episode_start_time = time.time()
        
        self.state = np.concatenate([np.zeros(48), self.action_list[-1]], axis=0)
        self.state = self.state.reshape(1,51)
        
        return self.state, self.info
    
    def get_reward(self, collision, action, geodesic, direction, energy):
        if collision != 0 and action[1] > 1e-5:
            return -50.0
        else:
            extreme_action_rwd_components = [
                1.0 if abs(action[0]/3.14) < 0.97 else 0.0,
                1.0 if abs(action[1]/2) < 0.97 else 0.0
            ]
            extreme_action_rwd = (extreme_action_rwd_components[0]*extreme_action_rwd_components[1])**.5
            geodesic_rwd = (1.0-math.tanh(geodesic))
            energy_rwd = (1.0 - energy/4.25)
            speed_rwd = self.world.player.get_velocity().length() / 2
        return (geodesic_rwd*speed_rwd*extreme_action_rwd*energy_rwd)**(1.0/4.0)
        
    def step(self, action):
        obs, rwd, done, truncated, info = self._step(action)
        return obs, rwd, done, truncated, info
    
    def _step(self, action):
        destination_success = 0
        self.world.tick(self.clock)
        self.render(self.render_mode)
        
        self.episode_step += 1
        
        done = truncated = False
        
        start_time = time.time_ns()
        
        local_actions, embedding, pedestrian_detected, episode_step_energy_consumption, wpt = self.initialAct()
        episode_step_latency = (time.time_ns() - start_time) / (10**9)
        
        # Preferred inference from local model
        if action == 0:
            mode = "Local"
            x_dir, y_dir = math.cos(local_actions[-1][0]), math.sin(local_actions[-1][0])
            self.controller._control.direction = carla.Vector3D(x_dir, y_dir, 0.0)
            self.controller._control.speed = local_actions[-1][1]
            self.local_step_counter += 1
        
        # Preferred inference from cloud model
        elif action == 1: 
            mode = "Cloud"
            start_time = time.time_ns()
            cloud_actions, cloud_energy_consumption = self.compute_cloud_policy(pedestrian_detected, self.eyelevel_rgb_array, wpt)
            cloud_latency = random.uniform(0, 0.2)
            latent_dir, latent_speed = np.random.normal(cloud_actions[0], cloud_latency), np.random.normal(cloud_actions[1], cloud_latency)
            x_dir, y_dir = math.cos(latent_dir), math.sin(latent_dir)
            self.controller._control.direction = carla.Vector3D(x_dir, y_dir, 0.0)
            self.controller._control.speed = latent_speed
            episode_step_energy_consumption += cloud_energy_consumption
            episode_step_latency += cloud_latency
            self.cloud_step_counter += 1
            
        self.world.player.apply_control(self.controller._control)
        
        obs = self._get_obs()
        
        self.distance_covered += obs[1] 
        
        reward  = self.get_reward(obs[0], self.action_list[-1], obs[1], obs[2], episode_step_energy_consumption)
        
        self.total_latency += episode_step_latency
        self.total_energy_consumption += episode_step_energy_consumption
        
        # Run Only for Eval
        if self.env_mode == "eval":
            self.addOverlayToVideo(mode)
        
        if obs[1] > 3:
            if self.env_mode == 'train':
                done = True
            else:
                pass
        
        if self.action_list[-1][1] > 1e-5:
            if obs[0] != 0.0:
                pass
            
        # Calculate % route completion
        for index in range(self.curr_wpt_index, min(self.curr_wpt_index+3, len(self.wpts))):
            wpt_coords = self.wpts[index]
            waypoint = self.world.map.get_waypoint(carla.Location(wpt_coords[0],wpt_coords[1],0.0), project_to_road=False, lane_type=carla.LaneType.Any)
            if waypoint:
                wpt_dir = waypoint.transform.get_forward_vector()
                wp_ego_veh = self.world.player.get_transform().location - carla.Location(wpt_coords[0],wpt_coords[1],0.0)

                dot_ve_wp = wp_ego_veh.x*wpt_dir.x + wp_ego_veh.y*wpt_dir.y + wp_ego_veh.z*wpt_dir.z
                # Good! Segment completed!
                if dot_ve_wp > 0:
                    # print("Waypoint ego_veh:",wp_ego_veh)
                    # print("Waypoint:",waypoint_coords)
                    # print("dot_ve_wp:",dot_ve_wp)
                    self.current_wp_index = index
                    self.percentage_route_completion = 100.0 * float(self._accum_meters[self.current_wp_index])/float(self._accum_meters[-1])
                else:
                    continue
        
        player_pos = self.world.player.get_location()
        
        curr_wpt = self.minimap.planner.get_closest_point(np.array([
            player_pos.x, player_pos.y
        ]))
        curr_wpt_x, curr_wpt_y = self.minimap.planner.path_pts[curr_wpt,:2]
        nxt_wpt = np.array([
            curr_wpt_x, curr_wpt_y
        ])
        player_coords = np.array([player_pos.x, player_pos.y])
        self.route_deviation = np.linalg.norm(player_coords - nxt_wpt)
        
        if self.route_deviation >= 3.0:
            truncated = True
        
        if self.state_manager.end_state:
            self.percentage_route_completion = 100.0
            destination_success = 1
            truncated = True
        
        if self.episode_step >= self.steps_per_episode:
            truncated = True
            
        if done or truncated:
            self.episode_count += 1
            
            print("Env lasts {} steps, restarting ...".format(self.episode_step))
            print("Task Metrics:")
            print("Collision Count:",self.num_pedestrian_infractions)
            print("Route Completion",(self.percentage_route_completion))
            episode_time = time.time() - self.episode_start_time
            
            self.task_metrics.append([
                destination_success,
                (self.percentage_route_completion*(0.5**(self.num_pedestrian_infractions/self.distance_covered))*(1-(self.total_energy_consumption/(1.65*(self.local_step_counter+self.cloud_step_counter))))*(0.8 if self.route_deviation >=1.5 else 1)), 
                self.percentage_route_completion*(0.5**(self.num_pedestrian_infractions/self.distance_covered))*(0.8 if self.route_deviation >=1.5 else 1),
                self.num_pedestrian_infractions/self.distance_covered,
                self.total_energy_consumption/self.distance_covered,
                1000/self.total_latency,
                self.percentage_route_completion, 
                self.route_deviation, 
                self.route_length*self.percentage_route_completion/100000.0, 
                episode_time, 
                self.local_step_counter, 
                self.cloud_step_counter])
            
        
        return np.concatenate([np.array(embedding), self.action_list[-1]], axis=0).reshape(1,51), reward, done, truncated, self.info
    
    def render(self, mode='human'):
        if mode == 'human':
            self.world.render(self.display)
            pygame.display.flip()
        else:
            pass
            
    
    def _get_actor_direction(self, actor, player_pos):
        actor_pos = actor.get_location()
        rotation = math.atan2(actor_pos.y-player_pos.y,actor_pos.x-player_pos.x)
        if abs(rotation) > (math.pi/4):
            return None
        return carla.Vector3D(math.cos(rotation),math.sin(rotation),0.0)
    
    def _get_actor_distance(self, actor, player_pos):
        return actor.get_location().distance(player_pos)
    
    
    def addOverlayToVideo(self, mode):
        # Add legends to video
        # mode = "Cloud (Latent)"
        mode_loc = (5, 15)
        infraction_loc = (5,40)
        energy_loc = (5,65)
        latency_loc = (5, 90)
        
        top_left = (2, 2)
        bottom_right = (200,100)
        alpha=0.45
        
        fontScale = 0.5
        color = (255,255,255)
        thickness = 1
        
        font=cv2.FONT_HERSHEY_COMPLEX
        
        # Create semi-transparent gray background
        overlay = self.eyelevel_rgb_array.copy()
        overlay = cv2.rectangle(overlay, top_left, bottom_right, (121, 125, 125), -1)  # A filled rectangle
        updated_img = cv2.addWeighted(overlay, alpha, self.eyelevel_rgb_array, 1 - alpha, 0)

        # Add mode
        updated_img = cv2.putText(updated_img, f'Mode:   {mode}', mode_loc, font,
                                    fontScale, color, thickness, cv2.LINE_AA)
        
        # Add Collision Count
        updated_img = cv2.putText(updated_img, f'CC:', infraction_loc, font,
                                    fontScale, color, thickness, cv2.LINE_AA)
        updated_img = cv2.putText(updated_img, f'{self.num_pedestrian_infractions:02.0f}', (84,40), font,
                                    fontScale, color, thickness, cv2.LINE_AA)
        
        # Add Energy 
        updated_img = cv2.putText(updated_img, f'Energy:  ', energy_loc, font,
                                    fontScale, color, thickness, cv2.LINE_AA)
        updated_img = cv2.rectangle(updated_img,(84,52), (184,68), (255,255,255), 1)
        updated_img = cv2.rectangle(updated_img,(84,52), (84+int(self.total_energy_consumption/25.0), 68), (0,255-self.total_energy_consumption/25.0,100+self.total_energy_consumption/25.0),cv2.FILLED)
        
        # Add Latency
        updated_img = cv2.putText(updated_img, f'Latency: ', latency_loc, font,
                                    fontScale, color, thickness, cv2.LINE_AA)
        updated_img = cv2.rectangle(updated_img,(84,77), (184,93), (255,255,255), 1)
        updated_img = cv2.rectangle(updated_img,(84,77), (84+int(self.total_latency/1.55), 93), (0,255-self.total_latency/1.55,100+self.total_latency/1.55),cv2.FILLED)
        
        self.rollout_video.write(updated_img)

        # Save minimap
        minimap_frame = np.flipud(np.rot90(pygame.surfarray.array3d(self.display)))
        minimap_frame = cv2.cvtColor(minimap_frame, cv2.COLOR_RGB2BGR)
        self.minimap_video.write(minimap_frame)
        
    
    def close(self):
        self.rollout_video.release()
        self.minimap_video.release()
        df = pd.DataFrame(self.task_metrics, columns=["Success Rate", "Ecological Navigation Score", "Navigation Score", "Infraction Count (/m)", "Energy Consumption (J/m)", "FPS (/s)","Route Completion (%)", "Avg. Route Deviation (m)", "Distance covered (km)", "Time taken (s)", "Number of steps taken by local model", "Number of steps taken by global model"])
        df.to_csv(self.task_metrics_path,index=False)

        self.client.apply_batch([carla.command.DestroyActor(x)
                           for x in self.vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for _id in self.sparse_id_dict['controllers']:
            self.world.world.get_actor(_id).stop()

        for _id in self.dense_id_dict['controllers']:
            self.world.world.get_actor(_id).stop()

        # print('\ndestroying %d walkers' % len(self.all_id_dict['walkers']))
        self.client.apply_batch([carla.command.DestroyActor(x)
                           for x in self.sparse_id_dict['controllers']])
        self.client.apply_batch([carla.command.DestroyActor(x)
                           for x in self.sparse_id_dict['walkers']])
        
        self.client.apply_batch([carla.command.DestroyActor(x)
                           for x in self.dense_id_dict['controllers']])
        self.client.apply_batch([carla.command.DestroyActor(x)
                           for x in self.dense_id_dict['walkers']])

        if self.world is not None:
            self.world.destroy()
        #self.world.world.apply_settings(self._settings)
        for camera in list(self.cameras.values()):
            camera.stop()
            camera.destroy()