# Overview
This model segments out ground that a wheeled robot could access. Based on the SegFormer B2 architecture. It avoids common obstacles like trees, bodies of water, chairs, tables, rocks, stairs etc... It will only segment ground that is reachable from the current POV, for example, it will not segment ground on an island surrounded by water. It also will not assume that ground is reachable if it cannot see a clear path from the current POV to that area, for example, if there is a patch of grass to the right of some tree and the only way to get to it would seemingly be by going behind the tree, it will not be segmented, as the model is not sure that the robot could traverse the area behind the tree to get to it.
# Run
The model is available [here](https://huggingface.co/JoshuaChick/TraversableGroundForRobot). It will be automatically downloaded when you run inference.py.

```git clone https://github.com/JoshuaChick/TraversableGroundForRobot```

```cd TraversableGroundForRobot```

```pip install -r requirements.txt```

```python inference.py```

or, if using Linux

```python3 inference.py```
# Example
![original](https://github.com/JoshuaChick/TraversableGroundForRobot/blob/main/readme_images/image.png?raw=true)
![result](https://github.com/JoshuaChick/TraversableGroundForRobot/blob/main/readme_images/result.png?raw=true)
