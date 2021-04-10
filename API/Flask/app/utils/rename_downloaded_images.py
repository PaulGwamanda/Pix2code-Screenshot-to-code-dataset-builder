import os

path = '501-550/'
i = 501
for filename in os.listdir(path):
    if (filename.find) != -1:
        if filename.endswith('.jpg' or '.jpeg'):
            os.rename(os.path.join(path, filename), os.path.join(path, str(i) + '.jpg'))
        elif filename.endswith('.png'):
            os.rename(os.path.join(path, filename), os.path.join(path, str(i) + '.png'))
        else:
            print("Parse Error, please upload a .jpg or .png file")
            continue
    i = i + 1

