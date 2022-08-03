# COAT Dataset Generation

This is the code used to generate the testing corpus of [COAT](https://proceedings.mlr.press/v162/xie22b.html) as described in the paper:

**[COAT: Measuring Object Compositionality in Emergent Representations](http://cs.stanford.edu/people/jcjohns/clevr/)**
 <br>
 <a href='https://siruixie.com'>Sirui Xie</a>,
 <a href='http://www.arimorcos.com/'>Ari Morcos</a>,
 <a href='http://www.stat.ucla.edu/~sczhu/'>Song-Chun Zhu</a>,
 <a href='https://vrama91.github.io/'>Ramakrishna Vedantam</a>
 <br>
 Presented at [ICML 2022](http://icml.cc/Conferences/2022)

It is developed based on the original repo of [CLEVR](https://github.com/facebookresearch/clevr-dataset-gen). You can use this code to render synthetic images and construct testing corpus. 

The testing corpus consists of tuples with strongly occluded multi-object scenes. They are obtained through a rejection sampling with certain occlusion threshold. The images in each tuple are correlated in a way that A and C have different sets of objects and different backgrounds, and B and D are the results of adding the same set of objects to A and C, respectively. 

<img src="../figures/coat_examples.png" alt="coat" width="100%" />

For image generation, we modified the original CLEVR generation code to introduce a colorful background and a spatially denser object distribution.

<img src="../figures/iid_examples.png" alt="iid" width="100%" />

We also introduce correlation between objects to go beyond the i.i.d. prior in objects. 

<img src="../figures/corr_examples.png" alt="corr" width="100%" />


## Step 1: Generating Images
First we render synthetic images using [Blender](https://www.blender.org/), outputting both rendered images as well as a JSON file containing ground-truth scene information for each image.

Blender ships with its own installation of Python which is used to execute scripts that interact with Blender; you'll need to add the `image_generation` directory to Python path of Blender's bundled Python. The easiest way to do this is by adding a `.pth` file to the `site-packages` directory of Blender's Python, like this:

```bash
echo $PWD/image_generation >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth
```

where `$BLENDER` is the directory where Blender is installed and `$VERSION` is your Blender version; for example on OSX you might run:

```bash
echo $PWD/image_generation >> /Applications/blender/blender.app/Contents/Resources/2.78/python/lib/python3.5/site-packages/clevr.pth
```

You can then render some images.

```bash
blender --background --python render_images.py
```

On OSX the `blender` binary is located inside the blender.app directory; for convenience you may want to
add the following alias to your `~/.bash_profile` file:

```bash
alias blender='/Applications/blender/blender.app/Contents/MacOS/blender'
```

If you have an NVIDIA GPU with CUDA installed then you can use the GPU to accelerate rendering like this:

```bash
blender --background --python render_images.py -- --use_gpu 1
```

To render images for testing corpus instead of training, use:
```bash
blender --background --python render_test_images.py -- --use_gpu 1
```
This program outputs scene images at `ROOT+'/images/'`, scene images with a different background at `ROOT+'/bgs/'`, object masks at `ROOT+'/masks/'`, ground-truth information for all objects at `ROOT+'/scenes/'`, and an index-to-object mapping for each scene at `ROOT+'/meta/'`. The method `render_subscene_obj` renders images of all possible object subsets for `scene_struct`. 


You can find [more details about image rendering here](image_generation/README.md).



## Step 2: Generating Testing Tuples
Next we generate testing tuples for the rendered images generated in the previous step.
This step takes as input the single JSON file containing all ground-truth scene information, and outputs a list of tuples of image paths.

You can generate testing tuples with `obj_algebra_test` in `test_generation.py`. Given the images for all possible subsets of objects in a scene, this function exhaustively searches through all valid object combinations of image A, B, C, D, rejects tuples with insufficient occlusion, and collects corresponding hard negatives: 

```python
   # convert masks to 0-1 value
   mask_A = np.array(Image.open(mask_A_path))[:,:,0]
   mask_A = np.where(mask_A==64, 0.0, mask_A)
   mask_A = np.where(mask_A==255, 1.0, mask_A)
   mask_B = np.array(Image.open(mask_B_path))[:,:,0]
   mask_B = np.where(mask_B==64, 0.0, mask_B)
   mask_B = np.where(mask_B==255, 1.0, mask_B)
   mask_C = np.array(Image.open(mask_C_path))[:,:,0]
   mask_C = np.where(mask_C==64, 0.0, mask_C)
   mask_C = np.where(mask_C==255, 1.0, mask_C)
   mask_D = np.array(Image.open(mask_D_path))[:,:,0]
   mask_D = np.where(mask_D==64, 0.0, mask_D)
   mask_D = np.where(mask_D==255, 1.0, mask_D)

   if np.abs(mask_A-mask_B+mask_C-mask_D).sum() > OCCLUSION_THRESHOLD: 
       # hard negative
       drop_idx_d = random.randint(0, len(subset_idx_list_D)-2)
       subset_idx_list_E = deepcopy(subset_idx_list_D)
       subset_idx_list_E =  subset_idx_list_E[:drop_idx_d]+list(subset_idx_list_E[drop_idx_d+1:])
       subset_idx_E = scene.objs2img["-".join( str(idx) for idx in subset_idx_list_E)]
       image_E_path = create_path(test_root, main_scene_idx, subset_idx_E, file_type="bgs")

       replace_idx_b = random.randint(0, len(subset_idx_list_B)-1)
       replace_idx_d = random.randint(0, len(subset_idx_list_D)-1)
       subset_idx_list_F = deepcopy(subset_idx_list_D)
       subset_idx_list_F[replace_idx_d] =  subset_idx_list_B[replace_idx_b]
       subset_idx_list_F = sorted(subset_idx_list_F)
       subset_idx_F = scene.objs2img["-".join( str(idx) for idx in subset_idx_list_F)]
       image_F_path = create_path(test_root, main_scene_idx, subset_idx_F, file_type="bgs")

       image_G_path = image_D_path.replace("/bgs/", "/color/")
       image_H_path = image_D_path.replace("/bgs/", "/material/")
       image_I_path = image_D_path.replace("/bgs/", "/shape/")
       image_J_path = image_D_path.replace("/bgs/", "/size/")

       tuples.append((image_A_path, image_B_path, image_C_path, image_D_path, image_E_path, image_F_path, image_G_path, image_H_path, image_I_path, image_J_path))
```


## Citation

```
@inproceedings{xie2022coat,
  title={COAT: Measuring Object Compositionality in Emergent Representations},
  author={Xie, Sirui and Morcos, Ari S and Zhu, Song-Chun and Vedantam, Ramakrishna},
  booktitle={International Conference on Machine Learning},
  pages={24388--24413},
  year={2022},
  organization={PMLR}
}
```
