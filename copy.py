import os

for dirt in ["images"]:
    for index in ['5']:
        source = "/checkpoint/siruixie/clevr_obj_test/output/"+index+"/obj_test/"+dirt+"/."
        dest = "/checkpoint/siruixie/clevr_obj_test/output/obj_test_occ/"+dirt
        os.system("cp -a "+source+" "+dest)
        print(dirt, index)
