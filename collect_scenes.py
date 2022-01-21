import json
all_scenes = []
for i in range(101, 901):
    print(i)
    scene_path = '/checkpoint/siruixie/clevr_obj_test/output/obj_test_final/scenes/CLEVR_new_%06d'%i+'.json'
    with open(scene_path, 'r') as f:
        all_scenes.append(json.load(f))
    output = {
        'info': {
            'date': "2022/01/20",
            'version': "1.0",
            'split': "new",
            'license': "Creative Commons Attribution (CC-BY 4.0)",
        },
        'scenes': all_scenes
    }
    with open('/checkpoint/siruixie/clevr_obj_test/output/obj_test_final/CLEVR_scenes.json', 'w') as f:
        json.dump(output, f)
