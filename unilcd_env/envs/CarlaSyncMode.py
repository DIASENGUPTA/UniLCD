import glob
import os
import sys
import numpy as np
from collections import defaultdict

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    import queue
except ImportError:
    import Queue as queue


import carla


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world,*sensors,**kwargs):
        # ,,,*sensors
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        
        # print(len(self._queues))
        return self
    def ret_q(self):
        self.frame=self.world.tick()
        return self._queues
    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data



class CarlaSyncMultiMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors,**kwargs):
        # ,,,*sensors
        self.world = world
        # print(sensors[0].values())
        # print(type(sensors))
        # print(sensors.keys())
        # raise Exception("breaking here")
        self.actor_ids = kwargs.get('actor_ids',[])
        self.sensors = {}
        for actor_id in self.actor_ids:
            self.sensors[actor_id] = list(sensors[0][actor_id].values())
        # print(self.sensors)
        # raise Exception("breaking here")
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = defaultdict(list)
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event, actor_id):
            q = queue.Queue()
            register_event(q.put)
            # print(self._queues[actor_id],type(self._queues[actor_id]))
            self._queues[actor_id].append(q)
                

        
        for actor_id in self.actor_ids:
            # print(f"sensor {actor_id}: ", self.sensors[actor_id])
            make_queue(self.world.on_tick, actor_id)
            for sensor in self.sensors[actor_id]:
                make_queue(sensor.listen, actor_id)
            # print(f"sensor queues for {actor_id}: ", self._queues[actor_id])
        return self
    
    def ret_q(self):
        self.frame=self.world.tick()
        return self._queues
    
    def tick(self, timeout):
        self.frame = self.world.tick()
        total_data = {}
        for actor_id in self.actor_ids:
            data = [self._retrieve_data(q, timeout) for q in self._queues[actor_id]]
            assert all(x.frame == self.frame for x in data)
            total_data[actor_id] = data
        return total_data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data