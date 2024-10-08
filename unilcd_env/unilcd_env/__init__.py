from gymnasium.envs.registration import register


register(
    id='unilcd_emb_env',
    entry_point='unilcd_env.envs:UniLCDEmbEnv',
    kwargs= {
        'steps_per_episode': 1500,
        'env_mode': 'eval',
        'render_mode': 'human',
        'width': 640,
        'height': 320,
        'host': '127.0.0.1',
        'port': 2000,
        'client_timeout': 30.0,
        'path': './unilcd_env/envs/path_points/path_points_t10_32_95.npy',
        'log_dir': './logs_dir/',
        'weather': 'WetCloudySunset',
        'tm_port': 6000,
        'world': 'Town10HD_Opt',
        'device': 0,
        'cloud_model_checkpoint': './unilcd_env/envs/il_model_checkpoints/cloud_model_ckpt.pth',
        'local_model_checkpoint': './unilcd_env/envs/il_model_checkpoints/local_model_ckpt.pth',
        'fps': 20,
        'rollout_video': './rollout_video.mp4',
        'minimap_video': './minimap_video.mp4',
        'task_metrics_path': './task_metrics.csv',
        'start_id': 95,
        'destination_id': 32,
        'debug': False,
        'gamma': 2.2,
        'filter': 'walker.pedestrian.0001',
        'rolename': 'hero',
    }    
)