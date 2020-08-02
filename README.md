# nlp
Dealing with the natural language processing of the new development goals

## Contribution guidelines
- We're using `requirements.txt` to keep track of dependency packages
- When setting up project on your local machine run the command to install the dependecy packages. It is recommended to use `venv` or `pipenv` on your local setup while working. (Do not commit the virtual environment files)
  - `pip install -r requirements.txt`
  - `pipenv install -r requirements.txt`
- If you're integrating a new package, remember to export your package dependencies using the. following command.
  - `pip freeze > requirements.txt`
