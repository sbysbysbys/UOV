from detectron2.data import MetadataCatalog
import numpy as np
import torch

def metadata_nuscenes():
    metadata = MetadataCatalog.get('nuscenes')
    metadata.set(
        thing_classes = ['barrier', 'bicycle', 'bus', 'car', 'construction vehicle', 'motorcycle', 'pedestrian', 'traffic cone', 'trailer', 'truck']
        , thing_colors = [
            (139, 69, 19),    # barrier brown
            (255, 255, 0),    # bicycle yellow
            (0, 255, 0),      # bus green
            (255, 0, 0),      # car red
            (255, 165, 0),    # construction_vehicle orange
            (0, 0, 255),      # motorcycle blue
            (255, 0, 203),  # pedestrian pink
            (255, 69, 0),     # traffic cone red-orange
            (128, 128, 128),  # trailer gray
            (0, 128, 0),      # truck dark green
            ]
        # , thing_dataset_id_to_contiguous_id={0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10}
        , thing_dataset_id_to_contiguous_id = {i: i for i in range(10)}
        , stuff_classes = ['barrier', 'bicycle', 'bus', 'car', 'construction vehicle', 'motorcycle', 'pedestrian', 'traffic cone', 'trailer', 'truck', 'driveable surface', 'other flat', 'sidewalk', 'terrain', 'manmade', 'vegetation', 'sky']
        , stuff_colors = [
            (139, 69, 19),    # barrier brown
            (255, 255, 0),    # bicycle yellow
            (0, 255, 0),      # bus green
            (255, 0, 0),      # car red
            (255, 165, 0),    # construction_vehicle orange
            (0, 0, 255),      # motorcycle blue
            (255, 192, 203),  # pedestrian pink
            (255, 69, 0),     # traffic cone red-orange
            (128, 128, 128),  # trailer gray
            (0, 128, 50),      # truck dark green
            (160, 160, 160),    # driveable surface
            (128, 128, 128),  # other flat
            (200, 200, 200),  # sidewalk
            (75, 75, 75),    # terrain
            (50, 50, 50),    # manmade
            (100, 100, 100),  # vegetation
            (225, 225, 225),  # sky
            ]
        , stuff_dataset_id_to_contiguous_id = {i: i for i in range(17)}
    )
    return metadata

def get_all_class_nuscenes():
    return ['barrier', 'bicycle', 'bus', 'car', 'construction vehicle', 'motorcycle', 'pedestrian', 'traffic cone', 'trailer', 'truck', 'driveable surface', 'other flat', 'sidewalk', 'terrain', 'manmade', 'vegetation', 'sky']

def get_stuff_classes():
    return [
    'animal',
    'human.pedestrian.adult',
    'human.pedestrian.child',
    'human.pedestrian.construction_worker',
    'human.pedestrian.personal_mobility',
    'human.pedestrian.police_officer',
    'human.pedestrian.stroller',
    'human.pedestrian.wheelchair',
    'movable_object.barrier',
    'movable_object.debris',
    'movable_object.pushable_pullable',
    'movable_object.trafficcone',
    'static_object.bicycle_rack',
    'vehicle.bicycle',
    'vehicle.bus.bendy',
    'vehicle.bus.rigid',
    'vehicle.car',
    'vehicle.construction',
    'vehicle.emergency.ambulance',
    'vehicle.emergency.police',
    'vehicle.motorcycle',
    'vehicle.trailer',
    'vehicle.truck',
    'flat.driveable_surface',
    'flat.other',
    'flat.sidewalk',
    'flat.terrain',
    'static.manmade',
    'static.other',
    'static.vegetation',
    'vehicle.ego'
    ]

def get_thing_classes():
    return [
    'animal',
    'human.pedestrian.adult',
    'human.pedestrian.child',
    'human.pedestrian.construction_worker',
    'human.pedestrian.personal_mobility',
    'human.pedestrian.police_officer',
    'human.pedestrian.stroller',
    'human.pedestrian.wheelchair',
    'movable_object.barrier',
    'movable_object.debris',
    'movable_object.pushable_pullable',
    'movable_object.trafficcone',
    'static_object.bicycle_rack',
    'vehicle.bicycle',
    'vehicle.bus.bendy',
    'vehicle.bus.rigid',
    'vehicle.car',
    'vehicle.construction',
    'vehicle.emergency.ambulance',
    'vehicle.emergency.police',
    'vehicle.motorcycle',
    'vehicle.trailer',
    'vehicle.truck'
    ]
def get_thing_colors():
    return [
        (125, 203, 171),
        (142, 174, 70),
        (13, 37, 214),
        (107, 130, 171),
        (46, 177, 97),
        (171, 109, 157),
        (25, 248, 50),
        (235, 246, 211),
        (255, 236, 49),
        (216, 58, 157),
        (140, 57, 108),
        (255, 151, 174),
        (52, 90, 3),
        (158, 18, 237),
        (220, 75, 84),
        (97, 220, 137),
        (181, 94, 50),
        (214, 90, 51),
        (217, 12, 146),
        (19, 229, 244),
        (91, 178, 197),
        (119, 212, 81),
        (74, 255, 243)
        ]
def get_stuff_colors():
    return [
        (125, 203, 171),
        (142, 174, 70),
        (43, 37, 214),
        (107, 130, 171),
        (46, 177, 97),
        (171, 109, 157),
        (25, 248, 50),
        (235, 246, 211),
        (255, 236, 49),
        (216, 58, 157),
        (140, 57, 108),
        (255, 151, 174),
        (52, 90, 3),
        (158, 18, 237),
        (220, 75, 84),
        (97, 220, 137),
        (181, 94, 50),
        (214, 90, 51),
        (217, 12, 146),
        (19, 229, 244),
        (91, 178, 197),
        (119, 212, 81),
        (74, 255, 243),
        (31, 31, 31),
        (62, 62, 62),
        (93, 93, 93),
        (124, 124, 124),
        (155, 155, 155),
        (186, 186, 186),
        (217, 217, 217),
        (235, 235, 235)
        ]
def metadata_nuscenes_2():
    metadata = MetadataCatalog.get('nuscenes')
    metadata.set(
        thing_classes = get_thing_classes(),
        thing_colors = get_thing_colors(),
        thing_dataset_id_to_contiguous_id = {i: i for i in range(23)},
        stuff_classes = get_stuff_classes(),
        stuff_colors = get_stuff_colors(),
        stuff_dataset_id_to_contiguous_id = {i: i for i in range(31)}
        )
    return metadata

def get_all_class_nuscenes_2():
    return get_stuff_classes()

class CategoryProcessor:
    def __init__(self):
        self.categories = [
    ('car', 'sedan', 'hatch-back', 'wagon', 'van', 'mini-van', 'SUV', 'jeep'),
    ('truck', 'pickup truck', 'lorry truck', 'semi truck', 'personal use truck', 'cargo hauling truck', 'trailer truck'),
    ('bendy bus', 'articulated bus', 'multi-section shuttle'),
    ('bus', 'standard bus', 'city bus', 'rigid bus'),
    ('construction vehicle', 'crane', 'bulldozer', 'dump truck'),
    ('motorcycle', 'a person on a motorcycle', 'scooter', 'vespa'),
    ('bicycle', 'road bicycle', 'mountain bike', 'electric bike', 'a person on a bicycle'),
    ('bicycle_rack', 'bicycle parking', 'bike rack', 'cycle stand'),
    ('vehicle trailer', 'truck trailer', 'car trailer', 'motorcycle trailer', 'container on a truck'),
    ('police vehicle', 'police car', 'police motorcycle', 'police bicycle'),
    ('ambulance', 'emergency ambulance', 'medical transport'),
    ('human', 'person', 'people', 'adult', 'walking adult', 'mannequin'),
    ('walking child', 'child'),
    ('construction worker', 'construction site worker'),
    ('stroller', 'baby stroller', 'child stroller', 'a stroller with a child'),
    ('wheelchair', 'manual wheelchair', 'a person on a manual wheelchair', 'electric wheelchair', 'a person on a electric wheelchair'),
    ('skateboard', 'a person on a skateboard', 'segway', 'a person on a segway','scooter','a person on a scooter'),
    ('police_officer', 'traffic police', 'patrolling officer'),
    ('animal', 'cat', 'dog', 'deer', 'bird'),
    ('cone', 'safety cone'),
    ('barrier', 'construction zone barrier', 'traffic barrier'),
    ('dolley', 'wheelbarrow', 'shopping cart','garbage-bin with wheels'),
    ('obstacle', 'full trash bag', 'construction material'),
    ('driveable surface', 'highway', 'cement road', 'asphalt road', 'road', 'gravel road', 'paved road', 'unpaved road'),
    ('walkway', 'sidewalk', 'bike path', 'traffic island'),
    ('grass', 'soil', 'sand', ' rolling hills', 'earth', 'ground level horizontal vegetation (< 20 cm tall)'),
    ('rail track', 'stairs with at most 3 steps', 'lake', 'river'),
    ('man-made object', 'building', 'house', 'premises', 'structure', 'part of construction', 'windowspane', 'door' ,'brick wall' ,'wall' ,'ceiling', 'guard rail', 'fence', 'pole', 'drainage', 'hydrant', 'flag', 'street sign', 'electric circuit box', 'parking meter', 'stairs with more than 3 steps', 'utility pole', 'signboard', 'road sign', 'traffic light', 'bus shelter', 'trash bin without wheels', 'fire hydrant', 'industrial building', 'chair', 'ladder', 'pavilion', 'plate', 'billboard', 'street lamp', 'bench', 'one side of the building', 'bridge','lamp', 'banner', 'generater', 'telephone booth', 'cell box', 'pavilion', 'tower', 'curtain', 'screen'),
    ('bush', 'tree', 'potted plant', 'plant'),
    ('undistinguishable object','sky', 'cloudy sky', 'clear sky', 'overcast'),
    ('bonnet under the picture',)
    ]

    def get_stuff_classes_dic(self):
        results = []
        for category in self.categories:
            category_list = [elem for elem in category]
            results += category_list
        return results
    
    def get_thing_classes_dic(self):
        results = []
        category_idx = 0
        for category in self.categories:
            if category_idx > 22:
                break
            category_idx += 1
            category_list = [elem for elem in category]
            results += category_list
        return results
    
    def get_stuff_colors_dic(self):
        stuff_colors = []
        category_idx = 0
        colors = get_stuff_colors()
        for category in self.categories:
            color = colors[category_idx]
            category_idx += 1
            for elem in category:
                stuff_colors.append(color)
        return stuff_colors
    
    def get_thing_colors_dic(self):
        thing_colors = []
        category_idx = 0
        colors = get_thing_colors()
        for category in self.categories:
            if category_idx > 22:
                break
            color = colors[category_idx]
            category_idx += 1
            for elem in category:
                thing_colors.append(color)
        return thing_colors

    def get_stuff_dataset_id_to_contiguous_id_dic(self):
        results = {}
        category_idx = 0
        elem_idx = 0
        for category in self.categories:
            for elem in category:
                results[elem_idx] = elem_idx
                elem_idx += 1
            category_idx += 1
        return results
    
    def get_thing_dataset_id_to_contiguous_id_dic(self):
        results = {}
        category_idx = 0
        elem_idx = 0
        for category in self.categories:
            if category_idx > 22:
                break
            for elem in category:
                results[elem_idx] = elem_idx
                elem_idx += 1
            category_idx += 1
        return results

    def get_stuff_dataset_id_to_class_id(self):
        results = {}
        category_idx = 0
        elem_idx = 0
        for category in self.categories:
            for elem in category:
                results[elem_idx] = category_idx
                elem_idx += 1
            category_idx += 1
        return results

    def get_class_id_to_real_class_id(self):
        return {
            0: 17, 1: 23, 2: 15, 3: 16, 4: 18, 5: 21, 6: 14, 7: 13, 8: 22, 9: 20, 10: 19,
            11: 2, 12: 3, 13: 4, 14: 7, 15: 8, 16: 5, 17: 6, 18: 1, 19: 12, 20: 9, 
            21: 11, 22: 10, 23: 24, 24: 26, 25: 27, 26: 25, 27: 28, 28: 30, 29: 29, 30: 31, 
        }

    def get_real_class_id_to_category_id(self):
        return {
            0: 0, 1: 0, 2: 7, 3: 7, 4: 7, 5: 0, 6: 7, 7: 0, 8: 0, 9: 1, 10: 0, 11: 0,
            12: 8, 13: 0, 14: 2, 15: 3, 16: 3, 17: 4, 18: 5, 19: 0, 20: 0, 21: 6, 22: 9,
            23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 0, 30: 16, 31: 0, -1: 0
        }


def metadata_nuscenes_dic():
    category_processor = CategoryProcessor()
    metadata = MetadataCatalog.get('nuscenes_dic')
    metadata.set(
        thing_classes = category_processor.get_thing_classes_dic(),
        thing_colors = category_processor.get_thing_colors_dic(),
        thing_dataset_id_to_contiguous_id = category_processor.get_thing_dataset_id_to_contiguous_id_dic(),
        stuff_classes = category_processor.get_stuff_classes_dic(),
        stuff_colors = category_processor.get_stuff_colors_dic(),
        stuff_dataset_id_to_contiguous_id = category_processor.get_stuff_dataset_id_to_contiguous_id_dic()
        )
    return metadata

def get_all_class_nuscenes_dic():
    category_processor = CategoryProcessor()
    return category_processor.get_stuff_classes_dic()

