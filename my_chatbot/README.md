## Files and Directories

- `data/` - Contains the data files used for training the chatbot.
  - `intents.json` - Contains the intents, patterns, and responses for the chatbot.
  - `responses.json` - Contains additional responses.
- `models/` - Contains the saved model and tokenizer.
  - `chatbot_model.h5` - The trained chatbot model.
  - `tokenizer.pickle` - The tokenizer used for processing text.
- `src/` - Contains the source code for the project.
  - `__init__.py` - Makes the directory a package.
  - `main.py` - The main script to run the chatbot.
  - `train.py` - Script to train the model.
  - `preprocess.py` - Script to preprocess the data.
  - `model.py` - Defines the model architecture.
  - `chatbot.py` - Contains the chatbot logic.
  - `utils.py` - Contains utility functions.
- `notebooks/` - Contains Jupyter notebooks for data preparation and model training.
  - `data_preparation.ipynb` - Notebook for preparing the data.
  - `model_training.ipynb` - Notebook for training the model.
- `requirements.txt` - Lists the dependencies required for the project.
- `README.md` - Provides an overview of the project.

## Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/Azavoloka7/my_chatbot.git
   cd my_chatbot
# my_chatbot
