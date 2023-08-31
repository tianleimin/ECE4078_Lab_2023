# Image to ground truth map generator
Credit Dr Calvin Vong and Kal Backman

## ❗WARNING❗
1. While the teaching team will try to maintain this code the best we can, this code is not guaranteed to be bug-free. 
It is the user's responsibility to verify that the generated map is correct, especially if you are testing with your own 
custom maps.

2. This code is ONLY to be used for testing purposes during the development of your code. IT IS CONSIDERED CHEATING if 
you are caught using this code to generate your ground truth maps for during any milestone marking.

## Usage
### Python version
This code has only been validated in later versions of python (3.9/10) onwards, so it will likely not work on the Ubuntu 18.04 which uses Python 3.6.9.

## Python Package Version differences
There are potentially package version clashes between the PenguinPi robot code and this Image-to-Map Generator, as a result we recommend you install the packages for this code in a separate python virtual environment.

This can be done with:
```bash
python -m venv map_generator
```
or
```bash
python3 -m pip map_generator
```
Depending on which installation of python you have installed on your system. You can then activate this environment with:

(On Windows)
```bash
map_generator\Scripts\activate
```
(On Mac or Linux)
```bash
source ./map_generator/bin/activate
```
From here you will install dependencies below.

### Install dependencies
All the dependencies are specified in [requirements.txt](requirements.txt). You may use this command to install the required dependencies:
```bash
python -m pip install -r requirements.txt
```
or
```bash
python3 -m pip install -r requirements.txt
```
Depending on which installation of python you have installed on your system.

### Configure the parameters
Change the parameters in [config.yaml](config.yaml) before running the code.

| Parameter         | notes                                                                                                                                                                 |
|-------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| input_image       | Name of the input image. All images should be placed in the [input_images](input_images) folder.                                                                      |
| output_map        | Name of the generated ground truth map. All maps will be generated in the [generated_ground_truth_maps](generated_ground_truth_maps) folder.                          |
| reference_map     | Name of the reference map. All reference maps should be placed in the [reference_maps](reference_maps) folder.                                                        |
| fruits            | List of fruit types to be detected. Append or remove items according to your map. Format - fruit_type: [B, G, R] (colour value of the fruit type to display)          |
| image_display_res | Resolution of the GUI to be displayed. This is only for display purposes and does not affect the generated map. Change this if the GUI does not fit your screen well. |
| max_click_radius  | Maximum radius of the click to be registered. Increase this if you are having trouble clicking on the objects.                                                        |
| arena_size        | Size of the arena in [m]. Change this if you are using your own custom arena                                                                                          |

It is recommended to keep the default values for the remaining parameters in the yaml file. However, you may change them if you wish to do so.

### Reference map

You may use the [reference_map_generator.py](reference_map_generator.py) to generate a reference map for your custom arena.

```bash
python3 reference_map_generator.py
```

### Run the code
Use this command to run the code:
```bash
python3 gt_map_generator.py
```

#### GUI controls
1. Drag the red boundary corners (top left box of the GUI) to the corners of the arena in the image.
2. Drag the objects to their respective positions in the image. The objects will appear as the colour specified in [config.yaml](config.yaml).
3. Click "save" at the bottom of the GUI to save the generated map.
4. To close the GUI, click usual "x" of the window

![gui_usage](docs/gui_usage.gif)

## Disclaimer
This code is provided as-is and has only been tested for the example use case provided here. It is not guaranteed to work for all use cases and may require modification to work for your use case. The code is provided for reference only and is not guaranteed to be the most efficient implementation.
