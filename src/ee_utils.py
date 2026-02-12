import ee
import geemap
from . import config

def initialize_ee(project_id=config.PROJECT_ID):
    """
    Authenticate and initialize Earth Engine.
    """
    try:
        ee.Initialize(project=project_id)
        print(f"Earth Engine initialized with project: {project_id}")
    except Exception as e:
        print(f"Error initializing Earth Engine: {e}")
        print("Attempting to authenticate...")
        ee.Authenticate(force=True)
        ee.Initialize(project=project_id)

def get_worldcover_dataset():
    """
    Load the ESA WorldCover dataset.
    """
    return ee.ImageCollection(config.EE_DATASET).first()

def create_interactive_map(dataset):
    """
    Create an interactive map with the dataset.
    """
    Map = geemap.Map()
    Map.add_basemap('SATELLITE')
    viz_params = {'bands': ['Map']}
    Map.addLayer(dataset, viz_params, "ESA WorldCover 2020")
    Map.add_colorbar(viz_params, label="Land Cover Class")
    return Map
