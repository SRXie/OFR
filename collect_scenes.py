import json
all_scenes = []
for i in range(1639):
    print(i)
    scene_path = '/checkpoint/siruixie/clevr_obj_test/output/obj_test_occ/scenes/CLEVR_new_%06d'%i+'.json'
    with open(scene_path, 'r') as f:
        all_scenes.append(json.load(f))
    output = {
        'info': {
            'date': "2021/12/27",
            'version': "1.0",
            'split': "new",
            'license': "Creative Commons Attribution (CC-BY 4.0)",
        },
        'scenes': all_scenes
    }
    with open('/checkpoint/siruixie/clevr_obj_test/output/obj_test_occ/CLEVR_scenes.json', 'w') as f:
        json.dump(output, f)
