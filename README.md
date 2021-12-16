# Super-Resolution

## Train
You can run this command to begin the training.
```
python main.py --epoch=5000 --batch_size=128
```

## Test with your images
You can put your images to **demo/images** directory and run this command to see the results:
```
python Demo.py --scale=2
```
You can modify **--scale** argument to your desired value, it affects to your output image size

**new_size = old_size * scale**
