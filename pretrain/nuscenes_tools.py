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

    # ('car', 'sedan', 'hatch-back', 'wagon', 'van', 'mini-van', 'SUV', 'jeep'),
    # ('truck', 'pickup truck', 'lorry truck', 'semi truck', 'personal use truck', 'cargo hauling truck', 'trailer truck'),
    # ('bendy_bus', 'articulated bus', 'multi-section shuttle'),
    # ('bus', 'standard bus', 'city bus', 'rigid bus'),
    # ('construction vehicle', 'crane', 'bulldozer', 'dump truck'),
    # ('motorcycle', 'a motorcycle with a person on it', 'scooter', 'vespa'),
    # ('bicycle', 'road bicycle', 'mountain bike', 'electric bike', 'a bicycle with a person on it'),
    # ('bicycle rack', 'bicycle parking', 'bike rack', 'cycle stand'),
    # ('vehicle trailer', 'truck trailer', 'car trailer', 'motorcycle trailer', 'container on a car'),
    # ('police vehicle', 'police car', 'police motorcycle', 'a police motorcycle with a police on it', 'police bicycle', 'a police bicycle with a police on it'),
    # ('ambulance', 'emergency ambulance', 'medical transport'),
    # ('human', 'person', 'peopel', 'adult', 'walking adult', 'mannequin'),
    # ('walking child', 'child'),
    # ('construction worker', 'construction site worker'),
    # ('stroller', 'baby stroller', 'child stroller', 'a stroller with a child in it'),
    # ('wheelchair', 'manual wheelchair', 'a wheelchair with a person on it', 'electric wheelchair'),
    # ('skateboard', 'a skateboard with a person on it', 'segway', 'a segway with a person on it','scooter','a scooter with a person on it'),
    # ('police_officer', 'traffic police', 'patrolling officer'),
    # ('animal', 'cat', 'dog', 'deer', 'bird'),
    # ('trafficcone', 'safety cone'),
    # ('barrier', 'construction zone barrier', 'traffic barrier'),
    # ('dolley', 'wheelbarrow', 'shopping cart','garbage-bin with wheels'),
    # ('obstacle', 'full trash bag', 'construction material'),
    # ('driveable_surface', 'highway', 'cement road', 'asphalt road', 'road'),
    # ('walkway', 'sidewalk', 'bike path', 'traffic island'),
    # ('grass', 'soil', 'sand', ' rolling hills', 'earth', 'gravel', 'ground level horizontal vegetation (< 20 cm tall)'),
    # ('rail track', 'stairs with at most 3 steps', 'lake', 'river'),
    # ('man-made object', 'building', 'wall', 'guard rail', 'fence', 'pole', 'drainage', 'hydrant', 'flag', 'street sign', 'electric circuit box', 'parking meter', 'stairs with more than 3 steps', 'utility pole', 'signboard', 'road sign', 'traffic light', 'bus shelter', 'trash bin without wheels', 'fire hydrant', 'industrial building', 'chair', 'ladder', 'pavilion', 'plate', 'billboard', 'street lamp', 'bench', 'one side of the building', 'building wall','bridge','lamp', 'banner', 'generater', 'telephone booth', 'cell box', 'pavilion', 'tower', 'curtain', 'screen', 'brick wall'),     # , 'door', 'window'
    # ('bush', 'tree', 'potted plant', 'plant'),
    # ('undistinguishable object','sky', 'cloudy sky', 'clear sky', 'overcast'),
    # ('ego car', 'bonnet under the picture')

    # 0('vehicle.car:Vehicle designed primarily for personal use', 'car', 'sedan', 'hatch-back', 'wagon', 'van', 'mini-van', 'SUV', 'jeep'),
    # 1('vehicle.truck:Vehicles primarily designed to haul cargo', 'truck', 'pickup truck', 'lorry truck', 'semi truck', 'personal use truck', 'cargo hauling truck', 'trailer truck'),
    # 2('vehicle.bendy_bus:Buses with two or more rigid sections linked by a pivoting joint', 'bendy_bus', 'articulated bus', 'multi-section shuttle'),
    # 3('vehicle.rigid_bus:Rigid bus and shuttle designed to carry more than 10 people', 'bus', 'standard bus', 'city bus', 'rigid bus'),
    # 4('vehicle.construction:Vehicle primarily designed for construction', 'construction vehicle', 'crane', 'bulldozer', 'dump truck'),
    # 5('vehicle.motorcycle:Gasoline or electric powered 2-wheeled vehicle', 'motorcycle', 'a motorcycle with a person on it', 'scooter', 'vespa'),
    # 6('vehicle.bicycle:Human or electric powered 2-wheeled vehicle', 'bicycle', 'road bicycle', 'mountain bike', 'electric bike', 'a bicycle with a person on it'),
    # 7('static_object.bicycle_rack:Area or device intended to park or secure the bicycles in a row', 'bicycle_rack', 'bicycle parking', 'bike rack', 'cycle stand'),
    # 8('vehicle.trailer:Any vehicle trailer', 'vehicle trailer', 'truck trailer', 'car trailer', 'motorcycle trailer', 'container on a car'),
    # 9('vehicle.police:All type of police vehicle', 'police vehicle', 'police car', 'police motorcycle', 'a police motorcycle with a police on it', 'police bicycle', 'a police bicycle with a police on it'),
    # 10('vehicle.ambulance:All types of ambulances', 'ambulance', 'emergency ambulance', 'medical transport'),
    # 11('human.pedestrian.adult:An adult pedestrian moving around the cityscape', 'human', 'person', 'peopel', 'adult', 'walking adult', 'mannequin'),
    # 12('human.pedestrian.child:A child pedestrian moving around the cityscape', 'walking child', 'child'),
    # 13('human.pedestrian.construction_worker:A human whose main purpose is construction work', 'construction worker', 'construction site worker'),
    # 14('human.pedestrian.stroller:Any stroller', 'stroller', 'baby stroller', 'child stroller', 'a stroller with a child in it'),
    # 15('human.pedestrian.wheelchair:Any type of wheelchair', 'wheelchair', 'manual wheelchair', 'a wheelchair with a person on it', 'electric wheelchair'),
    # 16('human.pedestrian.personal_mobility:Small electric or self-propelled vehicle', 'skateboard', 'a skateboard with a person on it', 'segway', 'a segway with a person on it','scooter','a scooter with a person on it'),
    # 17('human.pedestrian.police_officer:Any type of police officer', 'police_officer', 'traffic police', 'patrolling officer'),
    # 18('animal:All animals', 'animal', 'cat', 'dog', 'deer', 'bird'),
    # 19('movable_object.trafficcone:All types of traffic cones', 'trafficcone', 'safety cone'),
    # 20('movable_object.barrier:Any metal or concrete or water barrier temporarily placed in the scene in order to re-direct vehicle or pedestrian traffic', 'barrier', 'construction zone barrier', 'traffic barrier'),
    # 21('movable_object.pushable_pullable:Objects that a pedestrian may push or pull and not designed to carry humans', 'dolley', 'wheelbarrow', 'shopping cart','garbage-bin with wheels'),
    # 22('movable_object.debris:Debris or movable object too large to be driven over safely', 'obstacle', 'full trash bag', 'construction material'),
    # 23('flat.driveable_surface:Surfaces a car can drive on', 'driveable_surface', 'highway', 'cement road', 'asphalt road', 'road'),
    # 24('flat.sidewalk:paved path for pedestrians alongside a street', 'walkway', 'sidewalk', 'bike path', 'traffic island'),
    # 25('flat.terrain:Natural horizontal surfaces', 'grass', 'soil', 'sand', ' rolling hills', 'earth', 'gravel', 'ground level horizontal vegetation (< 20 cm tall)'),
    # 26('flat.other:Horizontal ground-level structures that do not belonging to other flat', 'rail track', 'stairs with at most 3 steps', 'lake', 'river'),
    # 27('static.manmade:Man-made structures', 'man-made object', 'building', 'wall', 'guard rail', 'fence', 'pole', 'drainage', 'hydrant', 'flag', 'street sign', 'electric circuit box', 'parking meter', 'stairs with more than 3 steps', 'utility pole', 'signboard', 'road sign', 'traffic light', 'bus shelter', 'trash bin without wheels', 'fire hydrant', 'industrial building', 'chair', 'ladder', 'pavilion', 'plate', 'billboard', 'street lamp', 'bench', 'one side of the building', 'building wall','bridge','lamp', 'banner', 'generater', 'telephone booth', 'cell box', 'pavilion', 'tower', 'curtain', 'screen', 'brick wall'),     # , 'door', 'window'
    # 28('static.vegetation:Vegetation higher than the ground (> 20cm)', 'bush', 'tree', 'potted plant', 'plant'),
    # 29('static.other:Background objects not matching other labels', 'undistinguishable object','sky', 'cloudy sky', 'clear sky', 'overcast'),
    # 30('vehicle.ego', 'ego car', 'bonnet under the picture')
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

    def text_same_class(self):
        same_class = torch.empty((0,0))
        for category in self.categories:
            class_size = len(category)
            same_class = torch.block_diag(same_class, torch.ones((class_size, class_size)))
        return same_class


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


def coloring_areas_bw(pano_seg, metadata, pano_seg_info=None, ids_to_colors=None):
    pano_seg_into255 = pano_seg.detach().cpu().numpy().astype(np.uint8)
    print(pano_seg_into255.size())
    max_pano_seg_into255 = np.max(pano_seg_into255)
    pano_seg_into255 = pano_seg_into255 * int(254 / max_pano_seg_into255)
    return pano_seg_into255

def coloring_areas(pano_seg, metadata, pano_seg_info, ids_to_colors, full_mask=False):
    # print("pano_seg_info = ", pano_seg_info)
    # print("ids_to_colors = ", ids_to_colors)
    _sinfo = {s["id"]: s for s in pano_seg_info} 
    pano_seg = pano_seg.detach().cpu()
    if full_mask:
        pano_seg = get_full_mask(pano_seg, _sinfo)
    segment_ids, areas = torch.unique(pano_seg, sorted=True, return_counts=True)
    areas = areas.numpy()
    sorted_idxs = np.argsort(-areas)
    _seg_ids = segment_ids[sorted_idxs]
    _seg_ids = _seg_ids.tolist()

    color_libary = set()
    for sid in _seg_ids:
        if sid in _sinfo:
            category_id = _sinfo[sid]["category_id"]
            id_number = 0
            while category_id*10+id_number not in ids_to_colors:
                id_number+=1
            _sinfo[sid]["color"] = [int(value * 255) for value in ids_to_colors[category_id*10+id_number]]
            while tuple(_sinfo[sid]["color"]) in color_libary:
                if 255 - min(_sinfo[sid]["color"]) > max(_sinfo[sid]["color"]):
                    min_value = min(_sinfo[sid]["color"])
                    min_index = _sinfo[sid]["color"].index(min_value)
                    _sinfo[sid]["color"][min_index] += 1
                else:
                    max_value = max(_sinfo[sid]["color"])
                    max_index = _sinfo[sid]["color"].index(max_value)
                    _sinfo[sid]["color"][max_index] += 1
            color_libary.add(tuple(_sinfo[sid]["color"]))
            _sinfo[sid]["cid_number"] = category_id*10+id_number
            del ids_to_colors[category_id*10+id_number]
            if _sinfo[sid]["isthing"]:
                _sinfo[sid]["categary"] = metadata.thing_classes[metadata.thing_dataset_id_to_contiguous_id[category_id]]
            else:
                _sinfo[sid]["categary"] = metadata.stuff_classes[metadata.stuff_dataset_id_to_contiguous_id[category_id]]

    # print("_sinfo = ", _sinfo)
    if len(_sinfo) > 0:
        color_index = np.vstack(([[0,0,0]], [_sinfo[idx]["color"] for idx in _sinfo]))
    else :
        color_index = np.array([[0, 0, 0]])
    pano_seg = np.array(pano_seg)
    color_map = color_index[pano_seg].astype(np.uint8)
    return color_map, _sinfo

def get_full_mask(pano_seg, _sinfo):
    mapped_ids = torch.tensor(list(_sinfo.keys()))

    # # single not_in_sinfo_id as 0
    # segment_ids, areas = torch.unique(pano_seg, sorted=True, return_counts=True)
    # not_in_mapped_ids_mask = ~torch.isin(segment_ids, mapped_ids)
    # not_in_mapped_id = torch.nonzero(not_in_mapped_ids_mask, as_tuple=False)
    # assert not_in_mapped_id.size(0) <= 1, ">1 ids corresponds to no labels. This is currently not supported"
    # not_in_mapped_id = not_in_mapped_id.item()
    # not_in_sinfo_positions = torch.nonzero(pano_seg == not_in_mapped_id, as_tuple=False)

    not_in_sinfo_mask = ~torch.isin(pano_seg, mapped_ids)
    not_in_sinfo_positions = not_in_sinfo_mask.nonzero(as_tuple=False)
    not_in_sinfo_set = {tuple(row.tolist()) for row in not_in_sinfo_positions}
    width = pano_seg.size(0)
    height = pano_seg.size(1)
    infect_value = 3

    while len(not_in_sinfo_set) != 0:
        in_sinfo_set = set()
        for pos in not_in_sinfo_set:
            if pos[0]+infect_value < width and pano_seg[pos[0]+infect_value, pos[1]].item() in _sinfo:
                pano_seg[pos[0], pos[1]] = pano_seg[pos[0]+infect_value, pos[1]]
                in_sinfo_set.add(pos)
            elif pos[0]-infect_value >= 0 and pano_seg[pos[0]-infect_value, pos[1]].item() in _sinfo:
                pano_seg[pos[0], pos[1]] = pano_seg[pos[0]-infect_value, pos[1]]
                in_sinfo_set.add(pos)
            elif pos[1]+infect_value < height and pano_seg[pos[0], pos[1]+infect_value].item() in _sinfo:
                pano_seg[pos[0], pos[1]] = pano_seg[pos[0], pos[1]+infect_value]
                in_sinfo_set.add(pos)
            elif pos[1]-infect_value >= 0 and pano_seg[pos[0], pos[1]-infect_value].item() in _sinfo:
                pano_seg[pos[0], pos[1]] = pano_seg[pos[0], pos[1]-infect_value]
                in_sinfo_set.add(pos)
        not_in_sinfo_set = not_in_sinfo_set.difference(in_sinfo_set)
        # print(len(not_in_sinfo_set))
        # import pdb; pdb.set_trace()
    return pano_seg



