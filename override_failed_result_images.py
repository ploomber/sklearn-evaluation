"""
Override failed result images so they become the new reference,
this happens sometimes when we make changes, we have to check manually
if differences are acceptable, then run this script
"""
from pathlib import Path
from glob import glob
import shutil

failed = glob('result_images/*/*-failed-diff*.png')

print('Found failed images...')

for f in failed:
    print('* ', f)


print('Getting reference images to replace...')


def get_ref_image(path):
    path = Path(path)
    return path.with_name(path.name.replace('-failed-diff', ''))

ref = [get_ref_image(f) for f in failed]


for f in ref:
    print('* ', f)

def get_new_location(path):
    path = Path(path)
    # get rid of the first part "result_images", and add the new relative
    # location inside tests/baseline_images
    return Path('tests', 'baseline_images', *path.parts[1:])

new_location = [get_new_location(f) for f in ref]

print('Copying...')

for old, new in zip(ref, new_location):
    print(f'*  {old} -> {new}')
    shutil.copy(old, new)