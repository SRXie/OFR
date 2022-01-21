import os

for dirt in [ "fgmasks"]: #["bgs", "color", "fgmasks", "images", "masks"]: #, "material", "meta","scenes", "shape",  "size"]:
    for index in range(2, 10):
        source = "/checkpoint/siruixie/clevr_obj_test/output/obj_test_occ_prep/"+str(index)+"/obj_test/"+dirt+"/."
        dest = "/checkpoint/siruixie/clevr_obj_test/output/obj_test_final/"+dirt
        os.system("cp -a "+source+" "+dest)
        print(dirt, index)
