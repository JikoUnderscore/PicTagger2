# PickTagger2

## Picture tagging program

I have a lot of picture that I want to tag but could not find a small and usefull software. See [notes](#notes)


## Requierments
- `winget install Python.Python.3.14` 
- `winget install OliverBetz.ExifTool`



## How to use (Windows)
run `python main.py` to generate a valid npz file, then run `python gui.py` and pick it.

```
python -m venv venv
venv\Scripts\activate.ps1
pip install requirements.txt
python main.py <path/to/folder/with/picture/with/subfolders>
python gui.py
```
if you have a probelems with `antelopev2` go to `~/.insightface\models\antelopev2\antelopev2` and move the file one directory back i.e., move them to `~/.insightface\models\antelopev2`



## Notes
- it works on my machine. dont care about yours
