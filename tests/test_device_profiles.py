from fedids_bench.evaluation.device_profiles import estimate_resource_costs, PROFILES

def test_device_profiles_estimation():
    for name in PROFILES.keys():
        est = estimate_resource_costs(
            model_num_params=5000,
            n_samples=100,
            n_epochs=5,
            profile_name=name
        )
        assert est["measurement_tag"] == "[ESTIMATED]"
        assert est["total_flops"] > 0
        assert est["estimated_energy_joules"] > 0
        assert "device_name" in est
