import os

PATH = "/data/fgolemo/intnet-bedrooms-png/"

files = [x for x in os.listdir(PATH) if "-d.png" in x[-6:] or "-rgb.png" in x[-8:]]
files.sort()
print (files[:10])

output = [f"{rgb} {d} 600\n" for d, rgb in zip(files[0::2], files[1::2])]

split_idx = int(round(len(output)*.8))
output_train = output[:split_idx]
output_test = output[split_idx:]

with open("interiornet_train_files_with_gt.txt", "w") as f:
    f.writelines(output_train)

with open("interiornet_test_files_with_gt.txt", "w") as f:
    f.writelines(output_test)


