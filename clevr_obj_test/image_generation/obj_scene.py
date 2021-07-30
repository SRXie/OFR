# Define Class of scene and obj
import itertools
from copy import deepcopy

# TODO: load it from properties.json
ATTR_GRAM = {
                "shape": {
                        "cube": "SmoothCube_v2",
                        "sphere": "Sphere",
                        "cylinder": "SmoothCylinder"
                    },
                "color": {
                        "gray": [87, 87, 87],
                        "red": [173, 35, 35],
                        "blue": [42, 75, 215],
                        "green": [29, 105, 20],
                        "brown": [129, 74, 25],
                        "purple": [129, 38, 192],
                        "cyan": [41, 208, 208],
                        "yellow": [255, 238, 51]
                    },
                "material": {
                        "rubber": "Rubber",
                        "metal": "MyMetal"
                    },
                "size": {
                        "large": 0.7,
                        "small": 0.35
                    }
                }

class Obj(object):
    def __init__(self, 
        index, 
        shape, 
        size, 
        scale, 
        material,
        threed_coords,
        rotation,
        pixel_coords,
        color,
        attrs2img=None,):

        self.index = index 
        self.shape = shape 
        self.size = size 
        self.scale = scale 
        self.material = material
        self.threed_coords = threed_coords
        self.rotation = rotation
        self.pixel_coords = pixel_coords
        self.color = color
        self.attrs2img = attrs2img
    
    def edit_attrs(self, attributes, return_imgs=False):
        ### Attrbutes is a list of attributes to be edited ###
        ### Check if it is the case first ##
        assert set(attributes).issubset(set(ATTR_GRAM.keys()))

        images = []
        objs = []
        attrs_enumeration = []
        # get enumeration of all attributes to be edited
        for attr in attributes:
            # get the current value of this attribute
            attr_val = eval('self.'+attr)
            # get all values in this attribute
            attr_domain = list(ATTR_GRAM[attr].keys())
            attr_domain.remove(attr_val)
            attrs_enumeration.append(attr_domain)

        # if only one attribute to edit, there is no combination
        if len(attributes) == 1:
            for attr_val in attrs_enumeration[0]:
                obj1 = deepcopy(self)
                attr_str = '"'+attr_val+'"'
                exec('obj1.'+attributes[0]+'='+ attr_str)
                objs.append(obj1)
                
                if return_imgs:
                    img_idx = self.attrs2img["-".join([str(obj1.size), str(obj1.color), str(obj1.material), str(obj1.shape)])]
                    images.append(img_idx)
            
            if return_imgs:
                return images
            else:
                return objs

        # a helper function to flatten nested tuples
        def flatten(object):
            gather = []
            for item in object:
                if isinstance(item, (list, tuple, set)):
                    gather.extend(flatten(item))            
                else:
                    gather.append(item)
            return gather    

        # a recurrsive function to compose attribute edits 
        def recurr_edit(prod, level):
            if level == 0:
                for cp in itertools.product(attrs_enumeration[level], prod):
                    obj1 = deepcopy(self)
                    cp = flatten(cp)
                    # print(prod)
                    # print(cp)
                    for i, attr in enumerate(attributes):
                        # print(cp[i])
                        attr_str = '"'+cp[i]+'"'
                        exec('obj1.'+attr+'='+ attr_str)
                    objs.append(obj1)

                    if return_imgs:
                        img_idx = self.attrs2img["-".join([str(obj1.size), str(obj1.color), str(obj1.material), str(obj1.shape)])]
                        images.append(img_idx)
                if return_imgs:
                    return images
                else:
                    return objs
            else: 
                prod = list(itertools.product(attrs_enumeration[level], prod))
                return recurr_edit(prod, level-1)

        level = len(attributes)
        prod = attrs_enumeration[level-1]
        return recurr_edit(prod, level-2)


class Scene(object):
    def __init__(self, 
        split, 
        image_index, 
        image_filename, 
        objects, 
        directions,
        objs_idx, 
        objs2img=None,
        bg_image_index=0):

        self.split = split
        self.image_index = image_index 
        self.image_filename = image_filename
        self.objects = objects
        self.directions = directions

        self.objs_idx = objs_idx
        self.objs2img = objs2img

        self.relationships = {}

        self.bg_image_index = bg_image_index

    def get_image(self):
        # TODO: get the image of the current scene
        return self.image_index

    def get_bg_image(self):
        return self.bg_image_index
    
    def get_obj_attrs(self):
        return set(ATTR_GRAM.keys())

    def to_dictionary(self):
        dictionary = deepcopy(self.__dict__)
        dictionary['objects'] = [obj.__dict__ for obj in self.objects]
        return dictionary
    
    def decompose(self, num_obj_decomp):
        image_pairs = []
        for obj_subset in itertools.combinations(set(self.objs_idx), num_obj_decomp):
            subset_idx_list_1 = sorted(list(obj_subset))
            subset_idx_list_2 = sorted(list(set(self.objs_idx).difference(set(obj_subset))))

            image_1 = self.objs2img["-".join( str(idx) for idx in subset_idx_list_1)]
            image_2 = self.objs2img["-".join(str(idx) for idx in subset_idx_list_2)]

            image_pairs.append((image_1, image_2)) # these are image indices

        return image_pairs

        

        