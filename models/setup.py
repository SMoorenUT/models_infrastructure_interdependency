from setuptools import setup, find_packages


setup(
    name="additional_models",
    entry_points={
        "movici.plugins": [
            f"intensity_capacity = additional_models.bridges_viaducts_IC:ICModel",
            f"safety_model = additional_models.bridges_viaducts_safety:SafetyModel",
            f"attributes = additional_models.attributes:BridgeViaductAttributes"
            
        ],
    },
    packages=find_packages(),
)
