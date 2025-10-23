# AI IMAGE CAPTIONER

------
## Table of Content

- [Description](#description)
- [Usage](#usage)
- [Demo and Links](#Demo-and-Links)
- [Deployed Link](#Deployed-Links)
- [Author/GitHub Repository](#AuthorCollaborators-GitHub-Repository)
- [License](#license)
------
## Set Up:

# 1) Install a supported Python alongside 3.14
brew install python@3.12

# 2) Make and activate a fresh virtual env with 3.12
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate

# 3) Install PyTorch into THIS env
python -m pip install --upgrade pip
pip install torch torchvision torchaudio  \
    --extra-index-url https://download.pytorch.org/whl/cpu

# 4) Verify
python -c "import torch,sys; print('torch', torch.__version__); print('python', sys.version)"


------
## Description:

Group repository for the Project we're making in the SDSU AI Club. An Image captioner AI that can given a photo and the AI will return a caption of the image to describe what's going on or what it sees.

------
## Usage:

To use this application, necessary dependencies will need to be installed. Run the following commands:

```aiignore
?
```

To run development, use:
```aiignore
?
```

------
## Demo and Links:
- Project is in the early stages of development, so no demo or links have been created.

------
## Deployed Links
- Unknown

------
## Author/Collaborators GitHub Repository:
- [Parsa Sedighi](https://github.com/Parsa-Sedighi)
- [Jayden Amjed](https://github.com/Jayden-Amjed)
- [Aaron Cayanan](https://github.com/aacayanan)
- [Wannida Polchan](https://github.com/wannidapolchan)
- [Charles Kim](https://github.com/kims1998)
- [Tony Munoz](https://github.com/mmunoz4538-wp)

-----
## License:
- A license hasn't been decided/chosen for this project.
