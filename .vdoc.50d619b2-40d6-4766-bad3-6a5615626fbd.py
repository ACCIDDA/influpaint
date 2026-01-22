# type: ignore
# flake8: noqa

scn_id = 868  # Choose your training scenario
experiment_name = "demo_jan2026"  # MLflow experiment name
scenario_spec = get_training_scenario(scn_id)
ddpm, dataset, transform, enrich, scaling_per_channel, data_mean, data_sd = (
    create_scenario_objects(
        scenario_spec, season_setup, image_size, channels, batch_size, epochs, device
    )



