## Deep Learning Toolkit (DLTK) docs

### Usage

Create a virtual environment and activate it:

```shell
python -m venv venv
source ./venv/bin/activate
```

Use pip to install the dependencies for the docs:
```shell
pip install -e .[doc]
```

Edit the source, clean the current build and rebuild html from source
```
make clean 
make html
```

Pop up a web browser and inspect the website:
```shell
python -m webbrowser build/html/index.html
```
